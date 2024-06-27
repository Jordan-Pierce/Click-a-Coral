import os
import random
import argparse

import shutil
import traceback

import numpy as np
import pandas as pd
from datetime import datetime as dt

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.metrics import auc
from from_Zooniverse import plot_samples
from reduce_annotations import get_image, remove_single_clusters


def group_annotations(pre, post, image_dir, output_dir, user, threshold):
    """
    This function groups annotations together by subject ID for the original annotations and
    the reduced annotations. It also calls other functions to visualize reductions and accuracy of annotations.

     Args:
         pre (Pandas dataframe): The dataframe containing all the original annotations
         post (Pandas dataframe): The dataframe containing the reduced annotations
         image_dir (str): The filepath to the image directory
         output_dir (str): The filepath to the output directory
         user (bool): Whether a user has been provided as an argument
         threshold (float): The IoU threshold value to determine mAP
    """
    # Make the subject ID's integers
    pre['Subject ID'] = pre['Subject ID'].astype(int)
    post['Subject ID'] = post['Subject ID'].astype(int)

    # Group both dataframes by subject ID
    pre = pre.groupby("Subject ID")
    post_groups = post.groupby("Subject ID")

    # Initialize the total amount of time spent on the annotations
    total_duration = pd.Timedelta(0, unit='s')

    # Initialize metrics dataframe
    metrics_df = pd.DataFrame(columns=['Subject ID', 'Average_IoU', 'Precision',
                                       'Recall', 'image_path', 'Number_of_Annotations', 'Mean_Average_Precision'])

    # Loop through the subject IDs in the original annotations
    for subject_id, subject_id_df in pre:

        print("Subject ID", subject_id)
        print("Subject Annotations", subject_id_df)

        # Get image information
        image_path, image_name, frame_name = get_image(subject_id_df.iloc[0], image_dir)

        # Get the reduced annotations for the subject ID
        if (post['Subject ID'] == subject_id).any():
            post_subject_id = post_groups.get_group(subject_id)

            # Get new dataframe
            no_singles = remove_single_clusters(subject_id_df)

            if not user:
                metrics_df = create_image_row(no_singles, subject_id, metrics_df, image_path, image_name, threshold)

            # Compare original annotations to reduction
            compare_pre_post(subject_id_df, post_subject_id, image_path, output_dir, image_name, user)

            # Compare the accuracy of original annotations to reduction
            compare_accuracy(no_singles, post_subject_id, image_path, output_dir, image_name)

            # Get the amount of time spent on one image
            time = get_time_duration(subject_id_df.iloc[0])
            total_duration += time
        else:
            continue

    return total_duration, metrics_df


def compare_pre_post(pre, post, image_path, output_dir, image_name, user):
    """
    Plots all the user annotations on the left and the reduced annotations on the right.

    Args:
        pre (Pandas dataframe): The original annotation dataframe
        post (Pandas dataframe: The reduced annotation dataframe
        image_path (str): A path to the image
        output_dir (str): The filepath to the output directory
        image_name (str): What the image should be saved as
        user (bool): Whether a user has been given or not
    """

    # Get a color mapping for all the users first
    usernames = pre['user_name'].unique().tolist()
    color_codes = {username: tuple(np.random.rand(3, )) for username in usernames}

    # Plot the images side by side
    image = plt.imread(image_path)
    fig, axs = plt.subplots(ncols=2, figsize=(20, 10), sharey=True)

    # Plot pre on the left subplot
    axs[0].imshow(image)
    axs[0].set_title(f'Original User Annotations: {len(pre)} annotations')
    for i, r in pre.iterrows():
        # Extract the values of this annotation
        x, y, w, h = r[['x', 'y', 'w', 'h']]

        # Check if a user has been provided
        if user:
            edge_color = "red"
        else:
            edge_color = color_codes[r['user_name']]

        # Create the figure
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=edge_color, facecolor='none')
        axs[0].add_patch(rect)

        # Plot the class label on the bbox
        axs[0].text(x + w * 0.02,
                    y + h * 0.98,
                    r['label'],
                    color='white', fontsize=8,
                    ha='left', va='top',
                    bbox=dict(facecolor=edge_color, alpha=0.5))

    # Plot image 2 on the right subplot
    axs[1].imshow(image)
    axs[1].set_title(f"Reduced Annotations: {len(post)} annotations")

    for i, r in post.iterrows():
        # Extract the values of this annotation
        x, y, w, h = r[['x', 'y', 'w', 'h']]
        y = float(y)

        # Create the figure
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='green', facecolor='none')
        axs[1].add_patch(rect)

        # Plot the class label on the bbox
        axs[1].text(x + w * 0.02,
                    y + h * 0.98,
                    r['label'],
                    color='white', fontsize=8,
                    ha='left', va='top',
                    bbox=dict(facecolor='green', alpha=0.5))

    # Save the figure
    plt.tight_layout()
    plt.savefig(f"{output_dir}\\User_vs_Reduction\\{image_name}", bbox_inches='tight')
    plt.close()


def compare_accuracy(pre, post, image_path, output_dir, image_name):
    """
    This function compares the accuracy of original annotations to that of the reduced annotations. It outputs
    a plot with the reduced annotations on the right and the original annotations colored in a range of red and green
    depending on the accuracy of the annotation to the reduced version.

    Args:
        pre (Pandas dataframe): The original annotation dataframe
        post (Pandas dataframe: The reduced annotation dataframe
        image_path (str): A path to the image
        output_dir (str): The filepath to the output directory
        image_name (str): What the image should be saved as
    """

    # Plot the images side by side
    image = plt.imread(image_path)
    fig, axs = plt.subplots(ncols=2, figsize=(20, 10), sharey=True)

    # Plot the original annotations on the left
    axs[0].imshow(image, cmap="RdYlGn")
    axs[0].set_title('Original User Annotations')

    # Color code the boxes based on the distance from reduced annotation
    color_code = mpl.colormaps['RdYlGn']
    bounds = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    norm = mpl.colors.BoundaryNorm(bounds, color_code.N, extend="both")

    if not pre.empty:

        # Plot pre on the left subplot
        for i, r in pre.iterrows():

            # Extract the values of this annotation
            x, y, w, h, d, iou, label = r[['x', 'y', 'w', 'h', "distance", "iou", "correct_label"]]

            # Color code the IoU or "accuracy"
            color = color_code(iou)

            # Create the figure
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
            axs[0].add_patch(rect)

            if label == 'Y':
                label_color = 'green'
            else:
                label_color = 'red'

            # Plot the class label on the bbox
            axs[0].text(x + w * 0.02,
                        y + h * 0.98,
                        r['label'],
                        color='white', fontsize=8,
                        ha='left', va='top',
                        bbox=dict(facecolor=label_color, alpha=0.5))

    # Add legend to the plot
    red_patch = patches.Patch(color='red', label='Incorrect')
    green_patch = patches.Patch(color='green', label='Correct')
    axs[0].legend(handles=[red_patch, green_patch], loc='upper right')

    # Plot reduced on the right subplot
    axs[1].imshow(image, cmap="RdYlGn")
    axs[1].set_title('Reduced Annotations')

    for i, r in post.iterrows():

        # Extract the values of this annotation
        x, y, w, h = r[['x', 'y', 'w', 'h']]
        y = float(y)

        # Create the figure
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='green', facecolor='none')
        axs[1].add_patch(rect)

        # Plot the class label on the bbox
        axs[1].text(x + w * 0.02,
                    y + h * 0.98,
                    r['label'],
                    color='white', fontsize=8,
                    ha='left', va='top',
                    bbox=dict(facecolor='green', alpha=0.5))

    plt.tight_layout()

    # Add the accuracy colorbar
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=color_code),
                        ax=axs, location="right", shrink=0.45, pad=0.01)
    cbar.set_label("Bounding Box IoU")

    # Save the figure to the output directory
    plt.savefig(f"{output_dir}\\Accuracy\\{image_name}", bbox_inches='tight')
    plt.close()


def plot_rankings(df, output_dir):
    """
    This function plots the top 100 users in each division based off of their mean average precision score.

    Args:
        df (Pandas dataframe): A dataframe containing information on all the users
        output_dir (str): The path to the output directory of where the plots should be saved
    """

    # Groups the users by division
    div_groups = df.groupby("Division")

    # String dict
    div_descriptions = {
        1: 'These users have made more than 100 annotations',
        2: 'These users have made up to 100 annotations',
        3: 'These users have made up to 50 annotations',
        4: 'These users have made up to 10 annotations'
    }

    # Loops through each division
    for div, div_df, in div_groups:

        # Subset and sort the dataframe for the top 100 users
        div_df.sort_values(by='Mean_Average_Precision', ascending=False, inplace=True)
        div_df = div_df.head(100)

        # Plot the user ranking
        plt.figure(figsize=(20, 10))
        plt.bar(div_df['User'], div_df['Mean_Average_Precision'])

        # Set axis labels
        plt.xlabel("Users")
        plt.ylabel("Mean Average Precision")
        plt.xticks(rotation=90, fontsize=5)

        # Set the title
        plt.suptitle(f"Division {div}: Top 100 Users", fontsize=20)
        plt.title(div_descriptions[div])

        plt.tight_layout()

        # Save the plot
        plt.savefig(f"{output_dir}\\division_{div}_ranking.jpg", bbox_inches='tight')
        plt.close()


def plot_point_graphs(user_df, image_df, output_dir):
    """
    This function plots the number of annotations on the (x-axis) and the mean average precision score (y-axis) for
    all users ond the left subplot and for all images on the right subplot.

    Args:
        user_df (Pandas dataframe): The user dataframe containing information for all users
        image_df (Pandas dataframe): The image dataframe containing information for all images
        output_dir (str): The path to where the plot should be stored
    """

    # Initialize the plot
    fig, axs = plt.subplots(ncols=2, figsize=(20, 10), sharey=True)

    # Plot the left subplot for users
    axs[0].scatter(user_df['Number_of_Annotations'], user_df['Mean_Average_Precision'])
    axs[0].set_ylim(0.0, 1.0)
    axs[0].set_title("User")

    # Plot right subplot for images
    axs[1].scatter(image_df['Number_of_Annotations'], image_df['Mean_Average_Precision'])
    axs[1].set_title("Images")

    # Set axis labels
    fig.supylabel("Mean Average Precision")
    fig.supxlabel("Number of Annotations")

    plt.tight_layout()

    # Save the plot
    plt.savefig(f"{output_dir}\\point_plot.jpg", bbox_inches='tight')
    plt.close()


def user_average(df, reduced_df, threshold):
    """
    This function creates a new dataframe with the average IoU and number of annotations for each user.

    Args:
        df (Pandas dataframe): The original dataframe
        reduced_df (Pandas dataframe): The reduced annotation dataframe
        threshold (float): The IoU threshold value to determine mAP
    """

    # Group by usernames
    users = df.groupby(['user_name'])

    # Initialize empty dataframe
    df = pd.DataFrame(columns=['User', 'Average_IoU', 'Number_of_Annotations', 'Mean_Average_Precision', 'Division'])

    # Loop through users
    for user, user_df in users:

        # Get the number of annotations and username
        length = len(user_df)
        user_name = user_df.iloc[0]['user_name']

        # Find the correct division
        if length < 10:
            div = 4
        elif length < 50:
            div = 3
        elif length < 100:
            div = 2
        else:
            div = 1

        # Remove single clusters
        user_df = remove_single_clusters(user_df)

        # Check if there are no annotations
        if user_df.empty:
            new_row = {'User': user_name, 'Average_IoU': 0,
                       'Number_of_Annotations': length, 'Mean_Average_Precision': 0, 'Division': div}
        else:
            # Find the average IoU and username
            average_iou = user_df['iou'].mean()

            # Find the mean average precision for the user
            ap = ap_for_user(user_df, reduced_df, threshold)

            # Add to dataframe
            new_row = {'User': user_name, 'Average_IoU': average_iou,
                       'Number_of_Annotations': length, 'Mean_Average_Precision': ap, 'Division': div}

        df = df._append(new_row, ignore_index=True)

    return df


def user_information(reduced, original, user, image_dir, output_dir, user_df, threshold):
    """
    This function looks at providing information regarding a specific user. It plots the same visualizations only for
    the user and creates a text file with information regarding the users status and ability.

    Args:
        reduced (Pandas dataframe): The reduced annotation dataframe
        original (Pandas dataframe): The original user annotation dataframe
        user (str): The user's name
        image_dir (str): The filepath to the image directory
        output_dir (str): The filepath to the output directory
        user_df (Pandas dataframe): Dataframe containing information on all users
        threshold (float): The IoU threshold value to determine mAP
    """

    # Finds the specific user
    user_info = user_df.loc[user_df['User'] == user]

    # Get the user info
    average_bbox_accuracy = user_info.iloc[0]['Average_IoU']
    length = user_info.iloc[0]['Number_of_Annotations']
    ap = user_info.iloc[0]['Mean_Average_Precision']
    div = user_info.iloc[0]['Division']

    # Isolates the division
    user_df = user_df[user_df['Division'] == div]

    # Sorts user dataframe
    user_df.sort_values(by='Mean_Average_Precision', ascending=False, inplace=True)
    user_df.reset_index(drop=True, inplace=True)
    user_info = user_df.loc[user_df['User'] == user]

    # Gets the users ranking
    ranking = user_info.index.tolist()

    # Set output directory for the user
    output_dir = f"{output_dir}\\{user}"

    # Make directory for the user
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}\\User_vs_Reduction", exist_ok=True)
    os.makedirs(f"{output_dir}\\Accuracy", exist_ok=True)

    # Find all the users annotations
    user_subset = original[original['user_name'] == user]

    # Find the subject IDS specific for the user
    subject_ids = user_subset['Subject ID'].unique().tolist()
    number_of_subject_ids = len(subject_ids)

    # Find the reduced annotations that correspond to the users
    reduced_subset = reduced[reduced['Subject ID'].isin(subject_ids)]

    # Run through the images for the user
    total_duration, images_df = group_annotations(user_subset, reduced_subset, image_dir, output_dir, True, threshold)

    # Get rid of the single clusters
    user_subset = remove_single_clusters(user_subset)

    # Create text file for user
    txt_file = open(f"{output_dir}\\{user}.txt", "w")

    txt_file.write(f"Username: {user}\n")
    txt_file.write(f"Time spent annotating: {total_duration}\n")
    txt_file.write(f"Total number of annotations: {length}\n")
    txt_file.write(f"Total number of images annotated: {number_of_subject_ids}\n")
    txt_file.write(f"IoU accuracy: {average_bbox_accuracy * 100}%\n")
    txt_file.write(f"Mean Average Precision: {ap * 100}%\n")
    txt_file.write(f"Your ranking is {ranking[0]} in division #{div}")

    txt_file.close()

    # Check if there is nothing to be plotted
    if user_subset.empty:
        return

    # Plot the user error by time
    # Map string values to color codes
    color_codes = {'Y': 'green', 'N': 'red'}
    color_values = [color_codes[value] for value in user_subset['correct_label']]

    # Make scatterplot
    plt.figure(figsize=(20, 10))
    plt.scatter(user_subset['created_at'], user_subset['iou'], c=color_values)

    # Set axis labels and title
    plt.ylabel("IoU")
    plt.xlabel("Time")
    plt.xticks(rotation=45)
    plt.title(f"Annotations over time for {user}")

    # Add legend outside the plot
    red_patch = patches.Patch(color='red', label='No')
    green_patch = patches.Patch(color='green', label='Yes')
    plt.legend(handles=[red_patch, green_patch], title="Correct Classification",
               loc='upper right')

    plt.tight_layout()

    # Save the plot
    plt.savefig(f"{output_dir}\\distance_distribution.jpg", bbox_inches='tight')
    plt.close()


def create_image_row(df, subject_id, images_df, image_path, image_name, threshold):
    """
    This function creates a new dataframe row with metrics on a specific image/Subject ID.

    Args:
        df (Pandas dataframe): A dataframe of annotations for a specific subject ID
        subject_id (int): The subject ID of an image
        images_df (Pandas dataframe): A dataframe that contains information on metrics for all the images
        image_path (str): Filepath to the image
        image_name (str): What the image should be called
        threshold (float): The IoU threshold value to determine mAP

    Returns:
        images_df (Pandas dataframe): The image dataframe with an additional row containing information on the image
        provided to this function
    """

    # Get the average IoU
    average_iou = df['iou'].mean()

    # Get length of the dataframe
    total = len(df)

    # Get the mAP, recall, and precision for an image
    ap, recall, precision = ap_for_image(df, threshold)

    # Create new row
    new_row = {'Average_IoU': average_iou, 'Precision': precision, 'Recall': recall,
               'Subject ID': subject_id, 'image_path': image_path, 'image_name': image_name,
               'Number_of_Annotations': total, 'Mean_Average_Precision': ap}

    # Add a new row to
    images_df = images_df._append(new_row, ignore_index=True)

    return images_df


def ap_for_image(df, threshold):
    """
    This function finds the mean average precision for an image.

    Args:
        df (Pandas dataframe): A dataframe containing all annotations for non-single clusters for a specific image
        threshold (float): The IoU threshold value to determine mAP

    Returns:
        ap (int): The mean average precision for the image
    """

    # Initialize arrays
    precision_array = [0.0]
    recall_array = [0.0]

    # Group by clusters
    df = df.groupby("clusters")

    # Loop through the clusters
    for cluster, cluster_df in df:

        # Get the precision and tp
        tp, precision, total = get_precision(cluster_df, threshold)

        # Get the recall for an object
        users = cluster_df['user_name'].unique().tolist()
        fn = 30 - len(users)
        recall = tp / (tp + fn)

        # Add to the precision and recall arrays
        precision_array.append(precision)
        recall_array.append(recall)

    # Turn arrays into dataframe to sort
    ap_df = pd.DataFrame({'Precision': precision_array, 'Recall': recall_array})
    ap_df.sort_values(by='Recall', inplace=True)

    # Find the mean recall for the image
    recall_mean = ap_df['Recall'].mean()
    precision_mean = ap_df['Precision'].mean()

    # Find the mean average precision for the image
    ap = auc(ap_df['Recall'], ap_df['Precision'])

    return ap, recall_mean, precision_mean


def ap_for_user(user_df, reduced_df, threshold):
    """
    This function finds the mean average precision for a user.

    Args:
        user_df (Pandas dataframe): A dataframe containing all annotations for non-single clusters for a specific user
        reduced_df (Pandas dataframe): The dataframe containing the reduced annotations
        threshold (float): The IoU threshold value to determine mAP

    Returns:
        ap (int): The mean average precision for the image
        """

    # Initialize arrays
    precision_array = [0.0]
    recall_array = [0.0]

    # Make the subject ID's integers
    user_df['Subject ID'] = user_df['Subject ID'].astype(int)
    reduced_df['Subject ID'] = reduced_df['Subject ID'].astype(int)

    # Group both dataframes by subject ID
    user_df = user_df.groupby("Subject ID")
    reduced_groups = reduced_df.groupby("Subject ID")

    # Loop through the images for a user
    for subject_id, subject_id_df in user_df:

        # Get the reduced annotations for the subject ID
        if (reduced_df['Subject ID'] == subject_id).any():
            reduced_subject_id = reduced_groups.get_group(subject_id)

            # Get the precision
            tp, precision, total = get_precision(subject_id_df, threshold)

            # Get number of object detections for user
            num_objects_found = len(subject_id_df['clusters'].unique().tolist())

            # Get the false negatives
            if len(reduced_subject_id) <= num_objects_found:
                fn = 0
            else:
                fn = len(reduced_subject_id) - num_objects_found

            # Get the recall
            if tp == 0:
                recall = 0
            else:
                recall = tp / (tp + fn)

            # Add to arrays
            precision_array.append(precision)
            recall_array.append(recall)
        else:
            continue

    # Sort arrays and find the mean average precision
    ap_df = pd.DataFrame({'Precision': precision_array, 'Recall': recall_array})
    ap_df.sort_values(by='Recall', inplace=True)
    ap = auc(ap_df['Recall'], ap_df['Precision'])

    return ap


def get_precision(df, threshold):
    """
    Find the precision for classification for a dataframe.

    Args:
        df (Pandas dataframe): A dataframe with original annotations
        threshold (float): The IoU threshold value to determine mAP

    Returns:
        tp (int): The true positive for a dataframe
        precision (int): The precision for correct classification
        total (int): The length of the dataframe
    """

    # Find the true positives
    yes_df = df[df['correct_label'] == "Y"]
    if yes_df.empty:
        tp = 0
    else:
        tp = len(yes_df[yes_df['iou'] > threshold])

    # Get the precision
    total = len(df)
    if total > 0:
        precision = tp / total
    else:
        precision = 0

    return tp, precision, total


def find_difficult_images(images_df, original_df, output_dir, n, image_dir):
    """
    This function finds the most and least difficult images for a dataframe containing information
    on a number of images.

    Args:
        df (Pandas dataframe): An image dataframe in the format provided from create_image_row
        output_dir (str): Path to the output directory
        n (int): The number of images to output
    """

    output_dir = f"{output_dir}\\Image Difficulty"

    # Make all the directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}\\Easy", exist_ok=True)
    os.makedirs(f"{output_dir}\\Hard", exist_ok=True)

    # Create text file for images
    #txt_file = open(f"{output_dir}\\hard_and_easy_images.txt", "w")

    # Sort the dataframe based on IoU
    images_df = images_df.sort_values(by='Mean_Average_Precision', ascending=False)

    # Add the top easiest images
    #txt_file.write(f"The top {n} easiest images for users were:\n")
    output_dir = f"{output_dir}\\Easy"
    for i, row in images_df.head(n).iterrows():

        subject_id = row['Subject ID']
        image_df = original_df[original_df['Subject ID'] == subject_id]
        #image_path, image_name, frame_name = get_image(image_df.iloc[0], image_dir)

        plot_image(image_df, row['image_path'], output_dir, row['image_name'])


        # txt_file.write(f"{row['image_name']}\n")
        # shutil.copy(row['image_path'], f"{output_dir}\\Easy")

    # Add the top hardest images
    #txt_file.write(f"The top {n} hardest images for users were:\n")
    output_dir = f"{output_dir}\\Hard"
    for i, row in images_df.tail(n).iterrows():
        subject_id = row['Subject ID']
        image_df = original_df[original_df['Subject ID'] == subject_id]
        # image_path, image_name, frame_name = get_image(image_df.iloc[0], image_dir)

        plot_image(image_df, row['image_path'], output_dir, row['image_name'])
        # txt_file.write(f"{row['image_name']}\n")
        # shutil.copy(row['image_path'], f"{output_dir}\\Hard")

    #txt_file.close()

def plot_image(df, image_path, output_dir, image_name):

    # Get a color mapping for all the users first
    usernames = df['user_name'].unique().tolist()
    color_codes = {username: tuple(np.random.rand(3, )) for username in usernames}

    # Plot the images side by side
    image = plt.imread(image_path)
    plt.figure(figsize=(20, 10))

    # Plot pre on the left subplot
    plt.imshow(image)
    plt.title(f'Original User Annotations: {len(df)} annotations')

    for i, r in df.iterrows():
        # Extract the values of this annotation
        x, y, w, h = r[['x', 'y', 'w', 'h']]

        edge_color = color_codes[r['user_name']]

        # Create the figure
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=edge_color, facecolor='none')
        plt.add_patch(rect)

        # Plot the class label on the bbox
        plt.text(x + w * 0.02,
                    y + h * 0.98,
                    r['label'],
                    color='white', fontsize=8,
                    ha='left', va='top',
                    bbox=dict(facecolor=edge_color, alpha=0.5))

    # Save the figure to the output directory
    plt.savefig(f"{output_dir}\\{image_name}", bbox_inches='tight')
    plt.close()


def get_time_duration(annotation):
    """
    This function gets the duration of time a user spent on an image.

    Args:
        annotation (Pandas dataframe row): The first annotation for an image by a user

    Returns:
        duration (Datetime object): The duration of time spent on the image
    """

    # Get the start and end time for the image
    start_time = annotation['started_at']
    end_time = annotation['finished_at']

    # Convert into datetime objects
    start_time = dt.strptime(start_time, '%Y-%m-%dT%H:%M:%S.%fZ')
    end_time = dt.strptime(end_time, '%Y-%m-%dT%H:%M:%S.%fZ')

    # Find the total duration
    duration = end_time - start_time

    return duration


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Visualize non-reduced and reduced annotations")

    parser.add_argument("--reduced_csv", type=str,
                        help="The CSV of reduced annotations")

    parser.add_argument("--extracted_csv", type=str,
                        help="The CSV of all annotations")

    parser.add_argument("--image_dir", type=str,
                        default='./Data',
                        help="The image directory")

    parser.add_argument("--output_dir", type=str,
                        default="./Reduced",
                        help="Output directory")

    parser.add_argument("--user", type=str, nargs='*',
                        default=None,
                        help="A list of usernames")

    parser.add_argument("--num_users", type=int,
                        default=None,
                        help="The number of random users to sample")

    parser.add_argument("--num_images", type=int,
                        default=10,
                        help="The number of easy/difficult images to output in the Image Difficulty folder")

    parser.add_argument("--iou_threshold", type=float,
                        default=0.6,
                        help="The IoU threshold to be taken into account for mAP")

    args = parser.parse_args()

    # Parse out arguments
    reduced_csv = args.reduced_csv
    extracted_csv = args.extracted_csv
    user_names = args.user
    image_dir = args.image_dir
    num_users = args.num_users
    num_images = args.num_images
    iou_threshold = args.iou_threshold

    # Turn both csv files into pandas dataframes
    pre = pd.read_csv(extracted_csv)
    post = pd.read_csv(reduced_csv)

    # Create output directories
    output_dir = f"{args.output_dir}\\Visualizations"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}\\User_vs_Reduction", exist_ok=True)
    os.makedirs(f"{output_dir}\\Accuracy", exist_ok=True)
    os.makedirs(f"{output_dir}\\Users", exist_ok=True)


    try:

        # Creates a dataframe for all users
        user_df = user_average(pre, post, iou_threshold)

        if user_names is None and num_users is None:
            total_duration, images_df = group_annotations(pre, post, image_dir, output_dir, False, iou_threshold)

            find_difficult_images(images_df, output_dir, num_images)

            plot_rankings(user_df, output_dir)
            plot_point_graphs(user_df, images_df, output_dir)

        else:
            output_dir = f"{output_dir}\\Users"

            if user_names is not None:

                for user_name in user_names:
                    user_information(post, pre, user_name, image_dir, output_dir, user_df, iou_threshold)

            else:
                usernames = pre['user_name'].unique().tolist()

                # Pull a random row
                user_names = random.sample(usernames, num_users)

                for user_name in user_names:
                    user_information(post, pre, user_name, image_dir, output_dir, user_df, iou_threshold)


        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")

        traceback.print_exc()


if __name__ == "__main__":
    main()
