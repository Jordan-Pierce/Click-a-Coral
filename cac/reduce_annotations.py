import os
import glob
import argparse
from time import time

import math
import traceback
import statistics

import datetime
import numpy as np
import pandas as pd
import random
from tqdm import tqdm

import torch
from torchvision import ops
import matplotlib.pyplot as plt

import cv2
import supervision as sv
from autodistill import helpers
from sklearn.cluster import OPTICS


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------

class AnnotationReducer:
    """
    A class to reduce annotations from Zooniverse data by clustering and filtering.
    
    This class processes annotation data, groups it by media ID and subject ID,
    performs clustering to identify similar annotations, reduces them to a single
    representative annotation, and exports the results in various formats.
    """
    
    def __init__(self, input_extracted_csv, media_dir, output_dir,
                 epsilon=70, cluster_plot_flag=False, cluster_size=3, big_box_threshold=0.3):
        """
        Initialize the AnnotationReducer with the specified parameters.
        
        Args:
            input_extracted_csv (str): The filepath to the csv containing the original annotations
            media_dir (str): The image directory with all the image files
            output_dir (str): The directory in which all transformations should be outputted
            epsilon (float, optional): The value to determine how far away two points should be to
                declare them as a cluster
            cluster_plot_flag (bool, optional): Whether the cluster plots should be saved
            cluster_size (int or float, optional): The minimum size of a cluster
            big_box_threshold (float, optional): The percent overlap of two boxes to consider one a 'big-boxer'
        """
        self.input_extracted_csv = input_extracted_csv
        self.media_dir = media_dir
        self.output_dir = output_dir
        self.epsilon = epsilon
        self.cluster_plot_flag = cluster_plot_flag
        self.cluster_size = cluster_size
        self.big_box_threshold = big_box_threshold
        
        # Timestamp for all files created in this run
        self.timestamp = self.get_now()
        
        # Create necessary directories
        os.makedirs(output_dir, exist_ok=True)
        self.output_yolo = f"{output_dir}/Yolo"
        os.makedirs(self.output_yolo, exist_ok=True)
        
        if self.cluster_plot_flag:
            os.makedirs(f"{output_dir}/Cluster Plots", exist_ok=True)
            
        # Output file paths
        self.output_reduced_csv = f"{output_dir}/reduced_annotations_{self.timestamp}.csv"
        self.output_extracted_csv = f"{output_dir}/extracted_data_w_additional_{self.timestamp}.csv"

    def process(self):
        """
        Main method to process all annotations and create the reduced dataset.
        """
        try:
            # Load and prepare data
            df = pd.read_csv(self.input_extracted_csv, index_col=0)
            # Drop Unnamed columns and unnecessary columns in one step
            cols_to_drop = [
                'classification_id', 
                'user_name', 
                'user_ip',
                'created_at', 
                'retired', 
                'user_id',
                'Unnamed: 0',
            ]
            df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

            # Convert Subject ID to int
            df['Subject ID'] = df['Subject ID'].astype(int)
            
            # Get class labels
            self.classes = df['label'].unique().tolist()
            
            # Group by Media ID
            media_groups = df.groupby("Media ID")
            
            # Process each media ID
            for media_id, media_id_df in tqdm(media_groups, desc="Processing Media IDs", unit="media"):
                self._process_media_id(media_id, media_id_df)
                        
            print("Processing complete!")
            return True
        
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
            return False
            
    def _process_media_id(self, media_id, media_id_df):
        """
        Process a single media ID group of annotations.
        
        Args:
            media_id: The media ID being processed
            media_id_df: DataFrame containing annotations for this media ID
            
        Returns:
            bool: True if processing was successful, False otherwise
        """
        # Initializes dicts for YOLO convert later
        detections = {}
        images = []
        
        # Group subset by Subject ID
        subject_ids = media_id_df.groupby("Subject ID")
        
        # Loops through Subject ID's with progress bar
        for subject_id, subject_id_df in tqdm(subject_ids, 
                                              desc=f"Processing Subject IDs for Media {media_id}", 
                                              unit="subject", leave=False):
            # Get image information
            image_path, image_name, image_frame = self._get_image(subject_id_df.iloc[0])
            
            # Skip if image path is empty (file not found)
            if not os.path.exists(image_path):
                raise Exception(f"Warning: Image path {image_path} does not exist. Skipping.")
                
            # If Subject ID has only one annotation, skip clustering and reduction
            if len(subject_id_df) >= self.cluster_size:
                self._process_subject_id(subject_id_df, image_path, image_name, image_frame, detections, images)
            else:
                print(f"Warning: Subject ID {subject_id}'s annotations are less than cluster size; skipping")
                
        # Process detected images if any
        if not images:
            return False
        else:
            # Creates YOLO directories
            yolo_dir = f"{self.output_yolo}/{media_id}"
            os.makedirs(yolo_dir, exist_ok=True)
            
            # Creates Detection Dataset
            ds = sv.DetectionDataset(classes=self.classes, images=images, annotations=detections)
            
            # Converts to YOLO format
            self._yolo_format(ds, yolo_dir)
            
            return True
        
    def _get_image(self, row):
        """
        Get information about an image from an annotation row.

        Args:
            row: A singular annotation

        Returns:
            frame_path: Filepath to the specific image
            image_name: A name for the image with the Media ID and Frame
            frame_name: The specific Frame Name of the image
        """
        # Get the meta
        media_id = int(row['Media ID'])
        frame_name = row['Frame Name']

        # Give name to the image
        image_name = f"{media_id} - {frame_name}"

        # Get media folder, the frame path
        media_folders = glob.glob(f"{self.media_dir}/*")
        media_folder = [f for f in media_folders if str(media_id) in f][0]
        frame_path = f"{media_folder}/frames/{frame_name}"

        if not os.path.exists(frame_path):
            raise Exception(f"Warning: Frame path {frame_path} does not exist. Trying with a different format.")
            frame_path = f"{media_folder}/{frame_name}/frames/{frame_name}"

        # Make sure the file exists before opening
        if not os.path.exists(frame_path):
            raise Exception(f"Warning: Frame path {frame_path} does not exist. Setting to empty string.")
            frame_path = ""

        return frame_path, image_name, frame_name
            
    def _process_subject_id(self, subject_id_df, image_path, image_name, image_frame, detections, images):
        """
        Process a single subject ID's annotations.
        
        Args:
            subject_id_df: DataFrame with annotations for this subject
            image_path: Path to the image file
            image_name: Name of the image
            image_frame: Frame name within the media
            detections: Dict to store detection results
            images: List to store image paths
        """            
        # Makes clusters and saves their labels
        cluster_labels, centers = self._make_clusters(subject_id_df, image_path, image_name)
        
        # Reduce bounding boxes based on clusters
        reduced_boxes, updated_annotations = self._reduce_boxes(subject_id_df, cluster_labels, centers)
        
        # Remove single clusters
        reduced_boxes = self._remove_single_clusters(reduced_boxes)
        
        # Remove big boxers
        reduced_boxes = self._remove_big_boxers(reduced_boxes)
        
        # Returns original annotations updated
        updated_annotations = self._update_annotations(updated_annotations, reduced_boxes)
        
        # Saves original annotations with cluster labels and distance to CSV
        self._save_to_csv(self.output_extracted_csv, updated_annotations)
        
        if reduced_boxes is None or reduced_boxes.empty:
            # Represents difficult image, skip
            return
        else:
            # Saves reduced annotations
            self._save_to_csv(self.output_reduced_csv, reduced_boxes)
            
            # Makes a detection based off of the reduced boxes and classes
            detection = self._make_detection(reduced_boxes)
            
            # Only add valid detections
            if detection is not None:
                # Adds detection to dict
                detections[image_path] = detection
                images.append(image_path)

    def _make_clusters(self, annotations, image_path, image_name):
        """
        Cluster annotations based on the OPTICS clustering algorithm.
        
        Args:
            annotations: Annotation dataframe corresponding to a subject ID
            image_path: Path to the image corresponding to a subject ID
            image_name: What the cluster plot should be named
            
        Returns:
            labels: List of clustering labels corresponding to the annotations
            centers: Dataframe containing the center points for the annotations
        """
        # Finds the center of each annotation bounding box
        centers = self._find_center(annotations)
        
        # Convert center values into usable array for OPTICS clustering
        array = centers.to_numpy()
        
        try:
            # Clusters using OPTICS
            clust = OPTICS(min_samples=int(self.cluster_size), 
                           cluster_method='dbscan', 
                           eps=self.epsilon, 
                           min_cluster_size=3)
            clust.fit(array)
        except Exception as e:
            raise Exception(f"Warning: Error in clustering: {e}")
            
        if len(clust.labels_) == 0:
            raise Exception("Warning: No clusters were found.")

        # Saves the clustering labels
        labels = clust.labels_
        
        # Plots the clustering if flag is set
        if self.cluster_plot_flag:
            self._plot_points(array, labels, image_path, image_name)
            
        return labels, centers

    def _find_center(self, df):
        """
        Find the center points from a dataframe of bounding box annotations.
        
        Args:
            df: The bounding box annotations
            
        Returns:
            centers: A dataframe with the x and y points for the center of each bounding box
        """
        # Vectorized calculation of center points
        centers = pd.DataFrame({
            'x_center': df['x'] + df['w'] / 2,
            'y_center': df['y'] + df['h'] / 2
        })
        
        return centers
        
    def _plot_points(self, array, labels, image_path, image_name):
        """
        Plot cluster points on an image.
        
        Args:
            array: Array of points to be plotted
            labels: Labels that correspond to the points
            image_path: Filepath to the image
            image_name: What the plot should be called
        """
        # Open the image
        image = plt.imread(image_path)
        
        # Set figure size
        plt.figure(figsize=(8, 6))
        
        # Plot the points
        scatter = plt.scatter(array[:, 0], array[:, 1], c=labels, s=50, alpha=0.6, cmap='viridis')
        
        # Add a legend and title
        legend1 = plt.legend(*scatter.legend_elements(), title="Clusters")
        plt.gca().add_artist(legend1)
        plt.title(f'OPTICS Clustering for {image_name}')
        
        # Plot and show the figure
        plt.grid(True)
        plt.imshow(image)
        
        # Save the cluster plot
        plt.savefig(f"{self.output_dir}/Cluster Plots/{image_name}", bbox_inches='tight')
        plt.close()

    def _reduce_boxes(self, annotations, labels, centers):
        """
        Reduce annotations to the bounding box of "best fit".
        
        Args:
            annotations: The original annotations to be reduced
            labels: Clustering labels
            centers: The center points for the annotations
            
        Returns:
            reduced: Reduced annotations
            annotations: The original annotations with bounds adjusted
        """
        # Add clusters as a column
        annotations = annotations.assign(clusters=labels)
        
        # Add centers to the annotations dataframe
        annotations.reset_index(drop=True, inplace=True)
        centers.reset_index(drop=True, inplace=True)
        annotations = pd.concat([annotations, centers], axis=1)
        
        # Group by clusters
        clusters = annotations.groupby('clusters')
        
        # Create reduced dataframe placeholder
        reduced = None
        
        # Iterate through the cluster groups
        for cluster, cluster_df in clusters:
            # Ignore if it has a cluster label of -1
            # This indicates that is a cluster of size 1
            if cluster == -1:
                # Check if cluster_df is not empty before assignment/concatenation
                if not cluster_df.empty:
                    if reduced is None:
                        reduced = cluster_df.copy()
                    else:
                        reduced = pd.concat([reduced, cluster_df], ignore_index=True)
                continue
            else:
                # Find the bounding box of best fit for the cluster
                avg_bbox = np.mean(cluster_df[['x', 'y', 'w', 'h']], axis=0)
                avg_bbox = pd.DataFrame(avg_bbox).T
                
                # Find the mode of all the labels
                mode = statistics.mode(cluster_df['label'])
                avg_bbox['label'] = mode
                
                new_row = cluster_df.iloc[0]
                new_row = pd.DataFrame(new_row).T
                # Remove the x, y, w, h, and label columns
                new_row = new_row.drop(columns=['x', 'y', 'w', 'h', 'label'])
                # Reset the indexes
                new_row.reset_index(drop=True, inplace=True)
                avg_bbox.reset_index(drop=True, inplace=True)
                
                # Create a new row with the context information and reduced bounding box
                # Ensure both DataFrames are not empty before concatenation
                if not new_row.empty and not avg_bbox.empty:
                    new_row = pd.concat([new_row, avg_bbox], axis=1)
                    # Add best-fit bounding box to reduced dataframe
                    reduced = pd.concat([reduced, new_row], ignore_index=True)
                    
        return reduced, annotations

    def _remove_single_clusters(self, df):
        """
        Remove single clusters from annotations after clustering.
        
        Args:
            df: Annotations post-clustering
            
        Returns:
            no_single_clusters: Annotations without single clusters
        """
        if df is None or df.empty:
            return df
            
        # Subset the dataframe to remove single clusters
        no_single_clusters = df[df['clusters'] != -1]
        no_single_clusters.reset_index(drop=True, inplace=True)
        
        return no_single_clusters

    def _remove_big_boxers(self, reduced):
        """
        Remove big boxer annotations from the reduced dataframe.
        
        Args:
            reduced: The reduced annotation dataframe
            
        Returns:
            reduced: The reduced annotation dataframe without big boxers
        """
        # Check if reduced is empty or has only one annotation
        if reduced is None or reduced.empty or len(reduced) <= 1:
            return reduced
            
        # Calculate the area for each bounding box in a vectorized way
        reduced['area'] = reduced['w'] * reduced['h']
        
        # Calculate mean area
        mean_area = reduced['area'].mean()
        
        # Find large boxes (potentially big-boxers)
        large_boxes = reduced[reduced['area'] > mean_area].copy()
        
        # If no large boxes, return the original dataframe
        if large_boxes.empty:
            return reduced
            
        # Get indices of boxes to remove
        to_remove = []
        
        for idx, row in large_boxes.iterrows():
            overlap_percent = self._total_overlap(row, reduced)
            if overlap_percent > self.big_box_threshold:
                to_remove.append(idx)
        
        # Remove the big boxer annotations
        if to_remove:
            return reduced.drop(to_remove)
        else:
            return reduced

    def _total_overlap(self, box, all_boxes):
        """
        Find the total area of overlap that a box has with all other boxes in the image.
        
        Args:
            box: The annotation that is being analyzed for an image
            all_boxes: All the annotations in an image
            
        Returns:
            percent: The amount of overlap between one box and all the other boxes in an image
        """
        # Skip if the dataframe is empty or has only one row
        if len(all_boxes) <= 1:
            return 0
        
        # Create a copy and filter out the current box
        # Using a different approach that doesn't rely on index equality
        if isinstance(box, pd.Series):
            # If box is a Series, we need to identify it by its values
            other_boxes = all_boxes.copy()
            # Create a boolean mask to find the exact row
            if 'x' in box and 'y' in box and 'w' in box and 'h' in box:
                mask = ((other_boxes['x'] == box['x']) & 
                        (other_boxes['y'] == box['y']) & 
                        (other_boxes['w'] == box['w']) & 
                        (other_boxes['h'] == box['h']))
                # Drop the row if found
                if mask.any():
                    other_boxes = other_boxes[~mask]
        else:
            # If box is not a Series, simply copy all_boxes
            other_boxes = all_boxes.copy()
        
        if len(other_boxes) == 0:
            return 0
        
        # Calculate intersection coordinates vectorized
        x1 = np.maximum(box['x'], other_boxes['x'].values)
        y1 = np.maximum(box['y'], other_boxes['y'].values)
        x2 = np.minimum(box['x'] + box['w'], other_boxes['x'].values + other_boxes['w'].values)
        y2 = np.minimum(box['y'] + box['h'], other_boxes['y'].values + other_boxes['h'].values)
        
        # Calculate areas where there is intersection
        widths = np.maximum(0, x2 - x1)
        heights = np.maximum(0, y2 - y1)
        intersection_areas = widths * heights
        
        # Sum all overlapping areas
        total_overlap_area = np.sum(intersection_areas)
        
        # Calculate percentage of overlap
        if total_overlap_area == 0:
            return 0
        else:
            return total_overlap_area / box['area']

    def _save_to_csv(self, output_csv, annotations):
        """
        Save annotations to a CSV file.
        
        Args:
            output_csv: Filepath for the output CSV file
            annotations: Annotations to be saved
        """
        # Checks if there are no annotations
        if annotations is None:
            return
        
        # Save the annotations to a csv file
        if os.path.isfile(output_csv):
            annotations.to_csv(output_csv, mode='a', header=False, index=False)
        else:
            annotations.to_csv(output_csv, index=False)

    def get_now(self):
        """
        Returns a datetime string formatted according to the current time.
        
        Returns:
            str: Timestamp string in format YYYY-MM-DD_HH-MM-SS
        """
        # Get the current datetime
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d_%H-%M-%S")
        
        return now

    def _make_detection(self, annotations):
        """
        Make a detection for an image.
        
        Args:
            annotations: Reduced annotations for an image
            
        Returns:
            detection: Detection dataclass for an image or None if annotations is empty
        """
        if annotations is None or annotations.empty:
            return None
        
        # Reset index for consistency
        annotations = annotations.reset_index(drop=True)
        
        # Calculate coordinates in a vectorized way
        x1 = annotations['x'].values
        y1 = annotations['y'].values
        x2 = x1 + annotations['w'].values
        y2 = y1 + annotations['h'].values
        
        # Create the xyxy array with coordinates
        xyxy = np.column_stack((x1, y1, x2, y2))
        
        # Create a mapping from class labels to indices
        class_to_idx = {label: idx for idx, label in enumerate(self.classes)}
        
        # Map labels to indices
        classes = np.array([class_to_idx[label] for label in annotations['label']])
        
        # Create a Detection dataclass for an image
        detection = sv.Detections(xyxy=xyxy, class_id=classes)
        
        return detection

    def _yolo_format(self, ds, yolo_dir):
        """
        Convert a DetectionDataset into YOLO format.
        
        Args:
            ds: A DetectionDataset for a single Media ID
            yolo_dir: Path to the YOLO directory for output
        """
        # Skip if dataset is None
        if ds is None:
            return
        
        # Create YOLO subdirectories
        yolo_images = f"{yolo_dir}/images"
        yolo_labels = f"{yolo_dir}/annotations"
        yolo_yaml = f"{yolo_dir}/data.yaml"
        
        os.makedirs(yolo_images, exist_ok=True)
        os.makedirs(yolo_labels, exist_ok=True)
        
        try:
            # Convert to YOLO format
            ds.as_yolo(images_directory_path=yolo_images,
                       annotations_directory_path=yolo_labels,
                       data_yaml_path=yolo_yaml)
            
            # Split to training and validation sets
            helpers.split_data(yolo_dir)
            
            # Create class txt file
            txt_path = f"{yolo_dir}/classes.txt"
            
            # Write class txt file
            with open(txt_path, "w") as file:
                for string in self.classes:
                    file.write(string + "\n")
        except TypeError as e:
            print(f"Warning: Error in YOLO conversion: {e}")
        except Exception as e:
            print(f"Warning: Unexpected error in YOLO conversion: {e}")
        finally:
            # Clean up directories if they exist but are empty
            try:
                if os.path.exists(yolo_images) and not os.listdir(yolo_images):
                    os.rmdir(yolo_images)
                if os.path.exists(yolo_labels) and not os.listdir(yolo_labels):
                    os.rmdir(yolo_labels)
            except Exception:
                pass  # Ignore cleanup errors

    def _update_annotations(self, pre, post):
        """
        Add additional columns to the original user annotation dataframe.
        
        This includes the distance from the reduced box, the IoU of the original annotation 
        and reduced box, the center point of the bounding box, and whether the classification 
        label of the original annotation is correct.
        
        Args:
            pre: Dataframe with original annotations
            post: Dataframe with reduced annotations
            
        Returns:
            updated_annotations: The updated original annotations
        """
        # Handle empty dataframes
        if pre is None or pre.empty:
            return pre
            
        if post is None or post.empty:
            return pre
            
        # Initialize dataframe to store updated annotations
        updated_annotations = pd.DataFrame()
        
        # Group original annotations by clusters
        pre_clusters = pre.groupby("clusters")
        
        # Process single clusters separately - check it's not empty before creating copy
        single_clusters = pd.DataFrame()
        if 'clusters' in pre.columns and (-1 in pre['clusters'].values):
            single_clusters = pre[pre['clusters'] == -1].copy()
        
        # Process each cluster group that's not a single cluster
        for cluster, pre_cluster_df in pre_clusters:
            if cluster == -1:
                continue
                
            # Find the reduced box that shares the same cluster ID
            reduced_box = post[post['clusters'] == cluster]
            
            # If no matching reduced box, mark as single cluster and continue
            if reduced_box.empty:
                pre_cluster_df = pre_cluster_df.copy()
                pre_cluster_df['clusters'] = -1
                if not pre_cluster_df.empty and not single_clusters.empty:
                    single_clusters = pd.concat([single_clusters, pre_cluster_df], ignore_index=True)
                elif not pre_cluster_df.empty:
                    single_clusters = pre_cluster_df.copy()
                continue
                
            # Get the dimensions for the reduced box
            reduced_box_row = reduced_box.iloc[0]
            x1, y1 = reduced_box_row['x'], reduced_box_row['y']
            x2, y2 = x1 + reduced_box_row['w'], y1 + reduced_box_row['h']
            reduced_dimensions = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float)
            
            # Calculate centers for reduced box
            reduced_center_x = x1 + reduced_box_row['w'] / 2
            reduced_center_y = y1 + reduced_box_row['h'] / 2
            reduced_box_label = reduced_box_row['label']
            
            # Calculate centers for all annotations in the cluster at once
            cluster_df = pre_cluster_df.copy()
            cluster_centers_x = cluster_df['x'] + cluster_df['w'] / 2
            cluster_centers_y = cluster_df['y'] + cluster_df['h'] / 2
            
            # Calculate distances using vectorized operations
            distances = np.sqrt((cluster_centers_x - reduced_center_x)**2 + (cluster_centers_y - reduced_center_y)**2)
            cluster_df['distance'] = distances
            
            # Determine if labels match the reduced box label
            cluster_df['correct_label'] = np.where(cluster_df['label'] == reduced_box_label, 'Y', 'N')
            
            # Calculate IoU for each box with the reduced box
            ious = []
            for _, box in cluster_df.iterrows():
                box_x1, box_y1 = box['x'], box['y']
                box_x2, box_y2 = box_x1 + box['w'], box_y1 + box['h']
                box_dimensions = torch.tensor([[box_x1, box_y1, box_x2, box_y2]], dtype=torch.float)
                iou = ops.box_iou(reduced_dimensions, box_dimensions).item()
                ious.append(iou)
                
            cluster_df['iou'] = ious
            
            # Add to updated annotations - check for emptiness before concatenation
            if not cluster_df.empty:
                if not updated_annotations.empty:
                    updated_annotations = pd.concat([updated_annotations, cluster_df], ignore_index=True)
                else:
                    updated_annotations = cluster_df.copy()
        
        # Add back the single clusters - check for emptiness before concatenation
        if not single_clusters.empty:
            if not updated_annotations.empty:
                updated_annotations = pd.concat([updated_annotations, single_clusters], ignore_index=True)
            else:
                updated_annotations = single_clusters.copy()
        
        return updated_annotations


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Reduce annotations for an image frame")

    parser.add_argument("--input_extracted_csv", type=str, required=True,
                        help="The input CSV file, extracted from the original annotations")

    parser.add_argument("--media_dir", type=str, required=True,
                        help="The image directory with all the image files for this season")

    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output directory for the reduced annotations")

    parser.add_argument("--epsilon", type=float,
                        default=70,
                        help="Epsilon value to be used for OPTICS clustering")

    parser.add_argument("--cluster_plot_flag", action="store_true",
                        help="Include if the cluster plots should be saved")

    parser.add_argument("--cluster_size", type=int,
                        default=3,
                        help="Determine the minimum size for a cluster")

    parser.add_argument("--big_box_threshold", type=float,
                        default=0.3,
                        help="The percent of overlap required to consider it a 'big boxer'")

    args = parser.parse_args()

    # Create an instance of the AnnotationReducer class
    reducer = AnnotationReducer(
        input_extracted_csv=args.input_extracted_csv,
        media_dir=args.media_dir,
        output_dir=args.output_dir,
        epsilon=args.epsilon,
        cluster_plot_flag=args.cluster_plot_flag,
        cluster_size=args.cluster_size,
        big_box_threshold=args.big_box_threshold,
    )
    
    print("Input CSV:", args.input_extracted_csv)
    print("Image Directory:", args.media_dir)
    print("Output Directory:", args.output_dir)
    
    print("Include Cluster Plots:", args.cluster_plot_flag)
    print("Epsilon:", args.epsilon)
    print("Cluster Size:", args.cluster_size)
    print("Big Box Percent Threshold:", args.big_box_threshold)

    try:
        # Process the annotations
        reducer.process()
        print("Done.")
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()