import os
import glob
import argparse

import math
import statistics
import numpy as np
import pandas as pd

import torch
from torchvision import ops
import matplotlib.pyplot as plt

import cv2
import supervision as sv
from sklearn.cluster import OPTICS


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------

def group_annotations(input_csv, num_samples, image_dir, output_dir, epsilon, cluster_plot_flag):
    """
        This function groups annotations in a dataframe according to their media ID and then
        subject ID. It also calls the other functions to reduce and save the annotations, as
        well as convert them to YOLO format.

        Args:
            input_csv (str): The filepath to the csv containing the original annotations
            num_samples (int): The number of media ID folders to sort through
            image_dir (str): The image directory with all the image files
            output_dir (str): The directory in which all transformations should be outputted
            epsilon (int): The value to determine how far away two points should be to
            declare them as a cluster
            cluster_plot_flag (bool): Whether the cluster plots should be saved
    """

    # Convert CSV to Pandas dataframe
    df = pd.read_csv(input_csv)
    df['Subject ID'] = df['Subject ID'].astype(int)

    # Initialize class labels
    classes = df['label'].unique().tolist()

    # Group by Media ID
    df = df.groupby("Media ID")

    # TODO: This will need to be gotten rid of
    # Initialize count
    #count = 0

    # Create output CSV file for reduced annotations
    output_csv = f"{output_dir}\\reduced_annotations.csv"

    # Create CSV file for original annotations
    updated_annotations_csv = f"{output_dir}\\extracted_data_w_additional.csv"

    # Loops through Media ID's
    for media_id, media_id_df in df:

        # Initializes dicts for YOLO convert later
        detections = {}
        images = {}

        # Group subset by Subject ID
        subject_ids = media_id_df.groupby("Subject ID")

        # Loops through Subject ID's
        for subject_id, subject_id_df in subject_ids:
            print("SubjectID", subject_id)
            print("Subject Annotations", subject_id_df)

            # Get image information
            image_path, image_name, image_frame = get_image(subject_id_df.iloc[0], image_dir)

            # If Subject ID has only one annotation, skip clustering and reduction
            if len(subject_id_df) > 1:

                # Makes clusters and saves their labels
                cluster_labels, centers = make_clusters(subject_id_df, epsilon, image_path, cluster_plot_flag, output_dir, image_name)

                # Reduce bounding boxes based on clusters
                reduced_boxes, updated_annotations = reduce_boxes(subject_id_df, cluster_labels, centers)

                # Remove single clusters
                reduced_boxes = remove_single_clusters(reduced_boxes)

                # Remove big boxers
                reduced_boxes = remove_big_boxers(reduced_boxes)

                # Returns original annotations updated
                updated_annotations = update_annotations(updated_annotations, reduced_boxes)

                # Saves original annotations with cluster labels and distance to CSV
                save_to_csv(updated_annotations_csv, updated_annotations, False)

                if reduced_boxes is None:
                    continue
                else:
                    # Saves reduced annotations
                    save_to_csv(output_csv, reduced_boxes, True)

                    # Makes a detection based off of the reduced boxes and classes
                    detection = make_detection(reduced_boxes, classes)

                    # Adds detection to dict
                    detections[image_frame] = detection

                    # Adds image to dict for YOLO convert
                    image = cv2.imread(image_path)
                    images[image_frame] = image

            else:
                continue

        # Checks to see if there are images in media folder
        if not images:
            continue
        else:
            # Creates YOLO directories
            yolo_dir = f"{output_dir}\\Yolo\\{media_id}"
            os.makedirs(yolo_dir, exist_ok=True)

            # Creates Detection Dataset
            ds = sv.DetectionDataset(classes=classes, images=images, annotations=detections)

            # Converts to YOLO format
            yolo_format(ds, yolo_dir, classes)

        #TODO: This will be deleted
       #Checks if Media ID count is over number of samples
        # count += 1
        # if count == num_samples:
        #     return


def make_clusters(annotations, epsilon, image_path, flag, output_dir, image_name):
    """
    This function clusters annotations based on the OPTICS clustering algorithm.

    Args:
        annotations (df): Annotation dataframe corresponding to a subject ID
        epsilon (int): Threshold value for how far apart points should be to consider
        them a cluster
        image_path (str): Path to the image corresponding to a subject ID
        flag (bool): Whether the cluster plots should be saved
        output_dir (str): The path to the output directory for the cluster plots
        image_name (str): What the cluster plot should be named

    Returns:
         labels (array): List of all the clustering labels corresponding to the annotations
         centers (Pandas dataframe): Dataframe containing the center points for the annotations
    """

    # Finds the center of each annotation bounding box
    centers = find_center(annotations)

    # Convert center values into usable array for OPTICS clustering
    array = centers.to_numpy()

    # Clusters using OPTICS
    clust = OPTICS(min_samples=0.0, cluster_method='dbscan', eps=epsilon, min_cluster_size=None)
    clust.fit(array)

    # Saves the clustering labels
    labels = clust.labels_

    # Plots the clustering
    if flag:
        plot_points(array, labels, image_path, output_dir, image_name)

    return labels, centers


def find_center(df):
    """
    This function find the center points from a dataframe of bounding box annotations

    Args:
        df (Pandas dataframe): The bounding box annotations

    Return:
        centers (Pandas dataframe): A dataframe with the x and y points for the center
        of each bounding box annotation
    """

    # Initialize dicts for x and y center points
    x_centers = []
    y_centers = []

    # Find the center of the bounding box for each annotation
    for i, row in df.iterrows():
        x_center = row['x'] + row['w'] / 2
        y_center = row['y'] + row['h'] / 2

        # Add values to dicts
        x_centers.append(x_center)
        y_centers.append(y_center)

    # Create centers dataframe
    centers = pd.DataFrame({'x_center': x_centers, 'y_center': y_centers})

    return centers


def plot_points(array, labels, image_path, output_dir, image_name):
    """
    This is a helper function to plot cluster points on an image

    Args:
        array (array): Array of points to be plotted
        labels (array): Labels that correspond to the points
        image_path (str): Filepath to the image
        output_dir (str): Path to where the plots should be saved
        image_name (str): What the plot should be called
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
    plt.savefig(f"{output_dir}\\Cluster Plots\\{image_name}", bbox_inches='tight')


def reduce_boxes(annotations, labels, centers):
    """
    This function reduces annotations to the bounding box of "best fit"

    Args:
        annotations (Pandas dataframe): The original annotations to be reduced
        labels (array): Clustering labels
        centers (Pandas dataframe): The center points for the annotations

    Return:
        reduced (Pandas dataframe): Reduced annotations
        annotations (Pandas dataframe): The original annotations with bounds adjusted
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
    columns = ['x', 'y', 'w', 'h', 'label']
    reduced = pd.DataFrame(columns=columns)

    # Iterate through the cluster groups
    for cluster, cluster_df in clusters:

        # Ignore if it has a cluster label of -1
        # This indicates that is a cluster of size 1
        if cluster == -1:

            # Add directly to reduced annotations
            reduced = pd.concat([cluster_df, reduced], ignore_index=True)
            continue

        else:

            # Find the bounding box of best fit for the cluster
            avg_bbox = np.mean(cluster_df[['x', 'y', 'w', 'h']], axis=0)
            avg_bbox = pd.DataFrame(avg_bbox).T

            # Find the mode of all the labels
            mode = statistics.mode(cluster_df['label'])
            avg_bbox['label'] = mode

            #TODO: This needs to be done better
            # Add in the missing columns
            # Get information from first row
            new_row = cluster_df.iloc[0]
            new_row = pd.DataFrame(new_row).T
            # Remove the x, y, w, h, and label columns
            new_row = new_row.drop(columns=['x', 'y', 'w', 'h', 'label'])
            # Reset the indexes
            new_row.reset_index(drop=True, inplace=True)
            avg_bbox.reset_index(drop=True, inplace=True)
            # Create a new row with the context information and reduced bounding box
            new_row = pd.concat([new_row, avg_bbox], axis=1)

            # Add best-fit bounding box to reduced dataframe
            reduced = pd.concat([new_row, reduced], ignore_index=True)

    return reduced, annotations


def remove_single_clusters(df):
    """
    This function removes single clusters from annotations after clustering

    Args:
        df (Pandas dataframe): Annotations post-clustering

    Returns:
        no_single_clusters (Pandas dataframe): Annotations without single clusters
    """

    # Initializes dataframe
    no_single_clusters = pd.DataFrame()
    count = 0

    # Iterates through annotations
    for i, row in df.iterrows():

        # Removes clusters of -1
        # This indicates a single cluster
        if row['clusters'] != -1:
            no_single_clusters = no_single_clusters._append(row, ignore_index=True)
        else:
            count += 1

    return no_single_clusters


def remove_big_boxers(reduced):
    """
    This function removes big boxer annotations from the reduced dataframe.

    Args:
        reduced (Pandas dataframe): The reduced annotation dataframe

    Returns:
        reduced (Pandas dataframe): The reduced annotation dataframe without big boxers
    """

    # Check if reduced is empty
    if reduced.empty:
        return

    # Find area for each bounding boxes
    reduced['area'] = reduced['h'] * reduced['w']

    # Find the mean area
    mean_area = reduced['area'].mean()

    # Sort dataframe by area
    reduced = reduced.sort_values(by='area', ascending=False)

    # Iterates through the rows of annotations to find cluster id of -1
    for i, row in reduced.iterrows():
        if row['area'] > mean_area:

            # The bbox may be a big boxer
            percent = total_overlap(row, reduced)

            # Remove boxes that have significant overlap
            if percent > 0.3:
                reduced = reduced.drop(i)
            else:
                continue

    return reduced


def total_overlap(box, all_boxes):
    """
    This function finds the total area of overlap that a box has with all other boxes in the image.

    Args:
        box (Pandas dataframe): The annotation that is being analyzed for an image
        all_boxes (Pandas dataframe): All the annotations in an image

    Returns:
        percent (int): The amount of overlap between one box and all the other boxes in an image
    """

    # Initialize variables
    total_overlap_area = 0
    num_overlaps = 0

    # Loop through all boxes in an image
    for i, other_box in all_boxes.iterrows():

        # Check to see if it is the same box
        if other_box.equals(box):
            continue

        else:
            # Calculate the coordinates of the intersection rectangle
            x1 = max(box['x'], other_box['x'])
            y1 = max(box['y'], other_box['y'])
            x2 = min(box['x'] + box['w'], other_box['x'] + other_box['w'])
            y2 = min(box['y'] + box['h'], other_box['y'] + other_box['h'])

            # If the intersection is valid (non-negative area), calculate the area
            if x1 < x2 and y1 < y2:
                overlap_area = (x2 - x1) * (y2 - y1)
                num_overlaps += 1
                total_overlap_area += overlap_area

    # Check if the box intersects with any other boxes
    if num_overlaps == 0:
        return 0

    else:
        # Calculate percentage
        percent = total_overlap_area / box['area']

        return percent


def get_image(row, image_dir):
    """
    This function gets information about an image from an annotation row

    Args:
        row (Pandas dataframe row): A singular annotation
        image_dir (str): Path to the image directory

    Returns:
        frame_path (str): Filepath to the specific image
        image_name (str): A name for the image with the Media ID and Frame
        frame_name (str): The specific Frame Name of the image
    """

    # Get the meta
    media_id = int(row['Media ID'])
    frame_name = row['Frame Name']

    # Give name to the image
    image_name = f"{media_id} - {frame_name}"

    # Get media folder, the frame path
    media_folders = glob.glob(f"{image_dir}\\*")
    media_folder = [f for f in media_folders if str(media_id) in f][0]
    frame_path = f"{media_folder}\\frames\\{frame_name}"

    return frame_path, image_name, frame_name


def save_to_csv(output_csv, annotations, column_drop):
    """
    This function saves annotations to a CSV file

    Args:
        output_csv (str): Filepath for the output CSV file
        annotations (Pandas dataframe): Annotations to be saved
        column_drop (bool): Drop additional columns, true or false
    """

    # Checks if there are no annotations
    if annotations is None:
        return

    # Checks if these columns need to be dropped
    if column_drop:
        # Get rid of unnecessary information
        annotations = annotations.drop(columns=['classification_id', 'user_name', 'user_ip', 'created_at', 'retired', 'user_id', 'Unnamed: 0'])

    # Save the annotations to a csv file
    if os.path.isfile(output_csv):
        annotations.to_csv(output_csv, mode='a', header=False, index=False)
    else:
        annotations.to_csv(output_csv, index=False)


def make_detection(annotations, class_labels):
    """
    This function makes a detection for an image.

    Args:
        annotations (Pandas dataframe): Reduced annotations for an image
        class_labels (list):  List of all the possible class labels

    Returns:
        detection (Detection): Detection dataclass for an image
    """

    # Resets index for annotations
    annotations.reset_index(inplace=True)

    # Initializes arrays for bounding boxes and classes
    xyxy = np.empty((len(annotations), 4))
    classes = np.empty(len(annotations), dtype=object)

    # Loops through annotations
    for i, row in annotations.iterrows():

        # Saves top left x, y value
        x1 = row['x']
        y1 = row['y']

        # Saves bottom right x, y value
        x2 = x1 + row['w']
        y2 = y1 + row['h']

        # Adds values to array
        xyxy[i] = [x1, y1, x2, y2]

        j = 0

        # Add the class label as a number to classes dict
        for cl in class_labels:
            if cl == row['label']:
                classes[i] = j
            j += 1

    # Create a Detection dataclass for an image
    detection = sv.Detections(xyxy=xyxy, class_id=classes)

    return detection


def yolo_format(ds, yolo_dir, classes):
    """
    This function converts a DetectionDataset into YOLO format.

    Args:
        ds (DetectionDataset): A DetectionDataset for a single Media ID
        yolo_dir (str): Path to the YOLO directory for output
        classes (list): List of classes
    """

    # Create YOLO subdirectories
    yolo_images = f"{yolo_dir}\\images"
    yolo_labels = f"{yolo_dir}\\labels"

    # Convert to YOLO format
    ds.as_yolo(images_directory_path=yolo_images, annotations_directory_path=yolo_labels)

    # Create class txt file
    txt_path = f"{yolo_dir}\\classes.txt"

    # Write class txt file
    with open(txt_path, "w") as file:
        for string in classes:
            file.write(string + "\n")


def update_annotations(pre, post):
    """
    This function adds additional columns to the original user annotation dataframe, including the distance from the
    reduced box, the IoU of the original annotation and reduced box, the center point of the bounding box, and whether
    the classification label of the original annotation is correct.

    Args:
        pre (Pandas dataframe): Dataframe with original annotations
        post (Pandas dataframe): Dataframe with reduced annotations

    Returns:
        updated_annotations (Pandas dataframe): The updated original annotations
    """

    # Initialize dataframe
    updated_annotations = pd.DataFrame()

    # Group original annotations by clusters
    pre_clusters = pre.groupby("clusters")

    # Initialize single clusters
    single_clusters = pd.DataFrame()

    # Loop through the clusters
    for cluster, pre_cluster_df in pre_clusters:
        # Skip for single clusters
        if cluster == -1:
            single_clusters = pre_cluster_df
            continue

        else:

            # Initialize arrays
            dist_array = []
            label_array = []
            iou_array = []

            # Find the centers of the reduced boxes that share the same cluster ID
            reduced_box = post.loc[post['clusters'] == cluster]

            # If there is no reduced box that matches -> add it to the single clusters
            if reduced_box.empty:
                pre_cluster_df['clusters'] = -1
                single_clusters = pd.concat([single_clusters, pre_cluster_df], ignore_index=True)
                continue
            else:

                # Get the dimensions for the reduced box
                x1 = reduced_box.iloc[0]['x']
                y1 = reduced_box.iloc[0]['y']
                x2 = x1 + reduced_box.iloc[0]['w']
                y2 = y1 + reduced_box.iloc[0]['h']

                reduced_dimensions = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float)

                # Find the center point and label for reduced box
                reduced_box_center = find_center(reduced_box)
                reduced_box_center = (reduced_box_center.iloc[0]['x_center'], reduced_box_center.iloc[0]['y_center'])
                reduced_box_label = reduced_box.iloc[0]['label']

                # Find the center values for each box in the original annotations
                cluster_centers = find_center(pre_cluster_df)

                # Loop through the centers for each original annotation
                for i, cluster_center in cluster_centers.iterrows():
                    cluster_center = (cluster_center['x_center'], cluster_center['y_center'])

                    # Find the distance between the 'best fit' box and an original annotation
                    distance = math.dist(reduced_box_center, cluster_center)

                    # Add to distance array
                    dist_array.append(distance)

                    # Annotation label
                    label = pre_cluster_df.iloc[i]['label']

                    # Determine if the label is correct
                    if label == reduced_box_label:
                        label_array.append('Y')
                    else:
                        label_array.append('N')

                    box = pre_cluster_df.iloc[i]

                    # Get the dimensions for the reduced box
                    x1 = box['x']
                    y1 = box['y']
                    x2 = x1 + box['w']
                    y2 = y1 + box['h']

                    box_dimensions = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float)

                    # Get the IoU
                    iou = ops.box_iou(reduced_dimensions, box_dimensions)
                    iou = iou.item()

                    # Add to iou array
                    iou_array.append(iou)

                # Add arrays as columns to the cluster
                pre_cluster_df = pre_cluster_df.assign(distance=dist_array)
                pre_cluster_df = pre_cluster_df.assign(correct_label=label_array)
                pre_cluster_df = pre_cluster_df.assign(iou=iou_array)

                # Add directly to original annotations
                updated_annotations = pd.concat([updated_annotations, pre_cluster_df], ignore_index=True)

    # Add back the single clusters to the original annotations
    updated_annotations = pd.concat([updated_annotations, single_clusters], ignore_index=True)

    return updated_annotations

# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Reduce annotations for an image frame")

    parser.add_argument("--csv", type=str,
                        help="Input CSV file")

    parser.add_argument("--image_dir", type=str,
                        default="./Data",
                        help="The image directory")

    parser.add_argument("--output_dir", type=str,
                        default="./",
                         help="Output directory")

    # TODO: Should get rid of this eventually
    parser.add_argument("--num_samples", type=int,
                        default=1,
                        help="Number of samples to run through")

    parser.add_argument("--epsilon", type=int,
                        default=70,
                        help="Epsilon value to be used for OPTICS clustering")

    parser.add_argument("--cluster_plot_flag", action="store_true",
                        help="Include if the cluster plots should be saved")


    args = parser.parse_args()

    # Parse out arguments
    input_csv = args.csv
    num_samples = args.num_samples
    epsilon = args.epsilon
    cluster_plot_flag = args.cluster_plot_flag

    # Set directories
    image_dir = args.image_dir
    output_dir = f"{args.output_dir}\\Reduced"

    # Make the output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}\\Yolo", exist_ok=True)

    if cluster_plot_flag:
        os.makedirs(f"{output_dir}\\Cluster Plots", exist_ok=True)

    print("Input CSV:", input_csv)
    print("Image Directory:", image_dir)
    print("Output Directory:", output_dir)
    print("Include Cluster Plots:", cluster_plot_flag)
    print("Epsilon:", epsilon)


    try:
        group_annotations(input_csv, num_samples, image_dir, output_dir, epsilon, cluster_plot_flag)
        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()