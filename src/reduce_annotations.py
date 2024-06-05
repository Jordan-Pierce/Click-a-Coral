import argparse
import sys
import glob
import os

import pandas as pd
import numpy as np
import math
import statistics

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.cluster import OPTICS
import cv2
import supervision as sv


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------

def group_annotations(input_csv, num_samples, image_dir, output_dir, epsilon):
    """
        This function groups annotations in a dataframe according to their media ID and then
        subject ID. It also calls the other functions to reduce and save the annotations, as
        well as convert them to YOLO format.

        Args:
            input_csv (str): The filepath to the csv containing the original annotations
            num_samples (int): The number of media ID folders to sort through
            image_dir (str): The image directory with all the image files
            output_dir (str): The directory in which all transformations should be outputted
            output_csv (str): The filepath to the reduced annotations csv file
            epsilon (int): The value to determine how far away two points should be to
            declare them as a cluster
        """

    # Convert CSV to Pandas dataframe
    df = pd.read_csv(input_csv)
    df['Subject ID'] = df['Subject ID'].astype(int)

    # Group by Media ID
    df = df.groupby("Media ID")

    # Initialize count and class labels
    count = 0
    classes = ['ANTIPATHESFURCATA', 'ANTIPATHESATLANTICA', 'BEBRYCESP', 'MADRACISSP', 'MADREPORASP',
               'MURICEAPENDULA', 'PARAMURICIADESP', 'SWIFTIAEXSERTA', 'THESEANIVEA']


    # Create output CSV file for reduced annotations
    output_csv = f"{output_dir}\\reduced_annotations.csv"

    #TODO: Note this is for testing purposes and in the final version only ONE CSV file should be created
    # Create output CSV file for reduced annotations WITHOUT single clusters
    without_single_csv = f"{output_dir}\\reduced_annotations_no_single_cluster.csv"

    # Create CSV file for original annotations
    original_annotations_csv = f"{output_dir}\\original_annotations.csv"

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
            image_path, image_name, jpg = get_image(subject_id_df.iloc[0], image_dir)

            # If Subject ID has only one annotation, skip clustering and reduction
            if len(subject_id_df) > 1:

                # Makes clusters and saves their labels
                labels = make_clusters(subject_id_df, epsilon, image_path)

                # Reduce bounding boxes based on clusters
                reduced_boxes, original_annotations = reduce_boxes(subject_id_df, labels)

                #TODO: Note that this is for testing purposes only
                # Final version should only output ONE reduction
                # Removes single clusters from reduced boxes
                no_single_clusters = remove_single_clusters(reduced_boxes)
                # Save the single cluster annotations
                save_to_csv(without_single_csv, no_single_clusters)

                #TODO: Note that this is for testing purposes only
                # Visually compare images with normal reductions to those with removed single clusters
                visual_compare(reduced_boxes, no_single_clusters, image_path, output_dir, image_name)

                # Returns original annotations with the distance from reduced bounding box
                original_annotations = calculate_distance(original_annotations, reduced_boxes)

                # Saves original annotations with cluster labels and distance to CSV
                save_to_csv(original_annotations_csv, original_annotations)

            else:

                # Skips reduction due to a singular annotation
                reduced_boxes = subject_id_df

                # Add in cluster label
                reduced_boxes.insert(14, 'clusters', -1)

                # Plot the single annotation
                plot_boxes(reduced_boxes, image_path, image_name, output_dir)

            # Saves reduced annotations to CSV
            save_to_csv(output_csv, reduced_boxes)

            # Makes a detection based off of the reduced boxes and classes
            detection = make_detection(reduced_boxes, classes)

            # Adds detection to dict
            detections[jpg] = detection

            # Adds image to dict for YOLO convert
            image = cv2.imread(image_path)
            images[jpg] = image

        # Creates YOLO directories
        os.makedirs(f"{output_dir}\\Yolo\\{media_id}", exist_ok=True)
        yolo_dir = f"{output_dir}\\Yolo\\{media_id}"

        # Creates Detection Dataset
        ds = sv.DetectionDataset(classes=classes, images=images, annotations=detections)

        # Converts to YOLO format
        yolo_format(ds, yolo_dir, classes)

        # Checks if Media ID count is over number of samples
        count += 1
        if count == num_samples:
            sys.exit(1)


def make_clusters(annotations, epsilon, image_path):
    """
    This function clusters annotations based on the OPTICS clustering algorithm.

    Args:
        annotations (df): Annotation dataframe corresponding to a subject ID
        epsilon (int): Threshold value for how far apart points should be to consider
        them a cluster
        image_path (str): Path to the image corresponding to a subject ID

    Returns:
         labels (array): List of all the clustering labels corresponding to the annotations
    """

    # Isolates x, y, width, and height values
    #annotations = annotations[['x', 'y', 'w', 'h']]

    # Finds the center of each annotation bounding box
    centers = find_center(annotations)

    # Convert center values into usable array for OPTICS clustering
    array = centers.to_numpy()

    # Clusters using OPTICS
    clust = OPTICS(min_samples=0.0, cluster_method='dbscan', eps=epsilon, min_cluster_size=None)
    clust.fit(array)

    # Saves the clustering labels
    labels = clust.labels_

    # Plots the clustering (OPTIONAL)
    #plot_points(array, labels, image_path)

    return labels


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


def plot_points(array, labels, image_path):
    """
    This is a helper function to plot cluster points on an image

    Args:
        array (array): Array of points to be plotted
        labels (array): Labels that correspond to the points
        image_path (str): Filepath to the image
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
    plt.title('OPTICS Clustering')

    # Plot and show the figure
    plt.grid(True)
    plt.imshow(image)
    plt.show()


def reduce_boxes(annotations, labels):
    """
    This function reduces annotations to the bounding box of "best fit"

    Args:
        annotations (Pandas dataframe): The original annotations to be reduced
        labels (array): Clustering labels

    Return:
        reduced (Pandas dataframe): Reduced annotations
        annotations (Pandas dataframe): The original annotations with bounds adjusted
    """

    # Add clusters as a column
    annotations = annotations.assign(clusters=labels)

    # Adjusts width and height so all bounding boxes are within the image
    annotations = stay_in_bounds(annotations)

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

    # Plot the reduced clustering (OPTIONAL)
    #plot_points(reduced.to_numpy(), reduced.index)

    return reduced, annotations


def stay_in_bounds(annotations):
    """
    This function adjusts the height and width of a bounding box if out of image bounds

    Args:
        annotations (Pandas dataframe): Annotations to be adjusted

    Returns:
        annotations (Pandas dataframe): Annotations WITH adjustments
    """

    # Loop through annotations
    for i, row in annotations.iterrows():

        # Checks if right corners of bounding box are outside the image width
        if row['x'] + row['w'] > row['Width']:
            row['w'] = (row['Width'] - row['x'])

        # Checks if bottom corners of bounding box are outside the image width
        if row['y'] + row['h'] > row['Height']:
            row['h'] = (row['Height'] - row['y'])

        # Checks if left corners of bounding box are outside the image width
        if row['x'] < 0:
            row['x'] = 0

        # Checks if top corners of bounding box are outside the image width
        if row['y'] < 0:
            row['y'] = 0

        # Resets the x, y, w, h values with adjustments
        annotations.loc[i] = row

    return annotations


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

    # Iterates through annotations
    for i, row in df.iterrows():

        # Removes clusters of -1
        # This indicates a single cluster
        if row['clusters'] != -1:
            no_single_clusters = no_single_clusters._append(row, ignore_index=True)

    return no_single_clusters

# TODO: These functions are currently not in use + may or may not become useful later on
# def remove_big_boxers(final_bbox):
#
#     # iterates through the rows of annotations to find cluster id of -1
#     for i, row in final_bbox.iterrows():
#         if row['clusters'] == -1:
#             # the bbox may be a big boxer
#             #print("row", row)
#             percent = total_overlap(row, final_bbox)
#             print("percent", percent)
#
#             # remove boxes that have significant overlap
#             if percent > 0.25:
#                 final_bbox.drop(final_bbox.index[i], inplace=True)
#         else:
#             continue
#     print("final", final_bbox)
#
#     return final_bbox
#
# def total_overlap(box, all_boxes):
#
#     total_overlap_area = 0
#     num_overlaps = 0
#     total_other_box_area = 0
#
#     for i, other_box in all_boxes.iterrows():
#         # test to see if it's the same as box
#         # print("other", other_box)
#         # print("box", box)
#         if other_box.equals(box):
#             print("samesies")
#             continue
#         else:
#
#             # Calculate the coordinates of the intersection rectangle
#             x1 = max(box['x'], other_box['x'])
#             y1 = max(box['y'], other_box['y'])
#             x2 = min(box['x'] + box['w'], other_box['x'] + other_box['w'])
#             y2 = min(box['y'] + box['h'], other_box['y'] + other_box['h'])
#
#             # If the intersection is valid (non-negative area), calculate the area
#             if x1 < x2 and y1 < y2:
#                 overlap_area = (x2 - x1) * (y2 - y1)
#                 num_overlaps += 1
#                 total_overlap_area += overlap_area
#
#                 # total area of the other boxes that are overlapping
#                 other_box_area = other_box['w'] * other_box['h']
#                 total_other_box_area += other_box_area
#
#     print("overlap area", total_overlap_area)
#     print("number of overlaps", num_overlaps)
#
#     if num_overlaps == 0:
#         return 0
#     else:
#         # calculate percentage
#         percent = total_overlap_area / total_other_box_area
#
#         return percent


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


def plot_boxes(df, image_path, image_name, output_dir):
    """
    This function plots the reduced annotations onto the image

    Args:
        df (Pandas dataframe): Reduced annotation dataframe
        image_path (str): The filepath for the image
        image_name (str): The name for the outputted image
        output_dir (str): Path to the output directory and where to save the image
    """

    # Open image
    image = plt.imread(image_path)

    # Loop through rows in dataframe
    for i, r in df.iterrows():

        # Extract the values of this annotation
        x, y, w, h = r[['x', 'y', 'w', 'h']]

        # Plot the boxes
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='black', facecolor='none')
        plt.gca().add_patch(rect)

        # Plot the class label on the bbox
        plt.text(x + w * 0.02,
                y + h * 0.98,
                r['label'],
                color='white', fontsize=8,
                ha='left', va='top',
                bbox=dict(facecolor='black', alpha=0.5))

    # Plot and save the image
    plt.title(f"{image_name}")
    plt.imshow(image)
    plt.savefig(f"{output_dir}\\Visual_Comparison\\{image_name}", bbox_inches='tight')


def save_to_csv(output_csv, annotations):
    """
    This function saves annotations to a CSV file

    Args:
        output_csv (str): Filepath for the output CSV file
        annotations (Pandas dataframe): Annotations to be saved
    """

    # Get rid of unnecessary information
    annotations = annotations.drop(columns=['classification_id', 'user_name', 'user_ip', 'created_at', 'retired', 'user_id', 'Unnamed: 0'])

    # Save the annotations to a csv file
    if os.path.isfile(output_csv):
        annotations.to_csv(output_csv, mode='a', header=False, index=False)
    else:
        annotations.to_csv(output_csv, index=False)


def visual_compare(annotations1, annotations2, image_path, output_dir, image_name):
    """
    This function creates a side by side plot to compare two sets of annotations. Note that this function was
    originally created to compare annotations with and without single clusters.

    Args:
        annotations1 (Pandas dataframe): The first set of annotations to be compared--with single clustering
        annotations2 (Pandas dataframe): The second set of annotations to be compared--without single clustering
        image_path (str): The filepath to the image
        output_dir (str): The path to the output directory
        image_name (str): The name of the image
    """

    # Open image
    image = plt.imread(image_path)

    # Create figure size and title
    plt.figure(figsize=(13, 7))
    plt.title(f"{image_name}")

    # Plot annotations1 on the left subplot
    plt.subplot(1, 2, 1)
    plt.imshow(image)

    # Loop through annotations
    for i, r in annotations1.iterrows():

        # Extract the values of this annotation
        x, y, w, h = r[['x', 'y', 'w', 'h']]

        # Create and plot bounding boxes
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='black', facecolor='none')
        plt.gca().add_patch(rect)

        # Plot the class label on the bbox
        plt.text(x + w * 0.02,
                y + h * 0.98,
                r['label'],
                color='white', fontsize=8,
                ha='left', va='top',
                bbox=dict(facecolor='black', alpha=0.5))
    plt.title('With Single Clusters')

    # Plot annotations2 on the right subplot
    plt.subplot(1, 2, 2)
    plt.imshow(image)

    # Loop through annotations
    for i, r in annotations2.iterrows():

        # Extract the values of this annotation
        x, y, w, h = r[['x', 'y', 'w', 'h']]

        # Create and plot bounding boxes
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='black', facecolor='none')
        plt.gca().add_patch(rect)

        # Plot the class label on the bbox
        plt.text(x + w * 0.02,
                y + h * 0.98,
                r['label'],
                color='white', fontsize=8,
                ha='left', va='top',
                bbox=dict(facecolor='black', alpha=0.5))
    plt.title('Without Single Clusters')

    # Save the plot
    plt.savefig(f"{output_dir}\\Visual_Comparison\\{image_name}", bbox_inches='tight')


def make_detection(annotations, class_labels):
    """
    This function makes a detection for an image

    Args:
        annotations (Pandas dataframe): Reduced annotations for an image
        class_labels (list):  List of all the possible class labels

    Returns:
        detection (Detection): Detection dataclass for an image
    """

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
    This function converts a DetectionDataset into YOLO format

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


def calculate_distance(pre, post):
    """
    This function calculates the distance between the original annotation and the "best fit" annotation

    Args:
        pre (Pandas dataframe): Dataframe with original annotations
        post (Pandas dataframe): Dataframe with reduced annotations

    Returns:
        annotations_with_distance (Pandas dataframe): The original annotations with the distance from reduced boxes
    """

    # Initialize dataframe
    annotations_with_distance = pd.DataFrame()

    # Group original annotations by clusters
    pre_clusters = pre.groupby("clusters")

    # Loop through the clusters
    for cluster, pre_cluster_df in pre_clusters:

        # Skip for single clusters
        if cluster == -1:
            single_clusters = pre_cluster_df
            continue

        else:

            # Initialize distance array
            dist_array = []

            # Find the centers of the reduced boxes that share the same cluster ID
            reduced_box = post.loc[post['clusters'] == cluster]
            reduced_box_center = find_center(reduced_box)
            reduced_box_center = (int(reduced_box_center['x_center']), int(reduced_box_center['y_center']))

            # Find the center values for each box in the original annotations
            cluster_centers = find_center(pre_cluster_df)

            # Loop through the centers for each original annotation
            for i, cluster_center in cluster_centers.iterrows():
                cluster_center = (int(cluster_center['x_center']), int(cluster_center['y_center']))

                # Find the distance between the 'best fit' box and an original annotation
                distance = math.dist(reduced_box_center, cluster_center)

                # Add to distance array
                dist_array.append(distance)

            # Add distance array as a column to the cluster
            pre_cluster_df = pre_cluster_df.assign(distance=dist_array)

            # Add directly to original annotations
            annotations_with_distance = pd.concat([annotations_with_distance, pre_cluster_df], ignore_index=True)

    # Add back the single clusters to the original annotations
    annotations_with_distance = pd.concat([annotations_with_distance, single_clusters], ignore_index=True)

    return annotations_with_distance

# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Reduce annotations for an image frame")

    parser.add_argument("-csv", type=str,
                        help="Input CSV file")

    parser.add_argument("-image_dir", type=str,
                        help="The image directory")

    parser.add_argument("-output_dir", type=str,
                         help="Output directory")

    parser.add_argument("-num_samples", type=int,
                        default=1,
                        help="Number of samples to run through")

    parser.add_argument("-season_num", type=int,
                        default=1,
                        help="Season number.")

    parser.add_argument("-epsilon", type=int,
                        default=100,
                        help="Epsilon value to be used for OPTICS clustering")


    args = parser.parse_args()

    # Parse out arguments
    season_num = args.season_num
    input_csv = args.csv
    num_samples = args.num_samples
    epsilon = args.epsilon

    # Set directories
    image_dir = f"{args.image_dir}\\Season_{season_num}"
    output_dir = f"{args.output_dir}\\Reduced"

    # Make the output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}\\Visual_Comparison", exist_ok=True)
    os.makedirs(f"{output_dir}\\Yolo", exist_ok=True)


    try:
        group_annotations(input_csv, num_samples, image_dir, output_dir, epsilon)

        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()