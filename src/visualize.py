import os
import sys
import random
import argparse

import shutil
import numpy as np
import pandas as pd
from datetime import datetime as dt

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.metrics import auc
from sklearn.linear_model import LinearRegression
from reduce_annotations import get_image, remove_single_clusters


def group_annotations(pre, post, image_dir, num_samples, output_dir, user):
    """
    This function groups annotations together by subject ID for the original annotations and
    the reduced annotations. It also calls other functions to visualize reductions and accuracy of annotations.

     Args:
         pre (Pandas dataframe): The dataframe containing all the original annotations
         post (Pandas dataframe): The dataframe containing the reduced annotations
         image_dir (str): The filepath to the image directory
         num_samples (int): The number of images/subject IDs to run through (soon to be irrelevant)
         output_dir (str): The filepath to the output directory
         user (bool): Whether a user has been provided in the arguments
    """
    # Make the subject ID's integers
    pre['Subject ID'] = pre['Subject ID'].astype(int)
    post['Subject ID'] = post['Subject ID'].astype(int)

    # Group both dataframes by subject ID
    pre = pre.groupby("Subject ID")
    post_groups = post.groupby("Subject ID")

    #TODO: Remove
    count = 0

    # Initialize the total amount of time spent on the annotations
    total_duration = pd.Timedelta(0, unit='s')
    # Initialize metrics dataframe
    metrics_df = pd.DataFrame(columns=['Subject ID', 'Average_IoU', 'Precision',
                                       'Recall', 'image_path', 'Number_of_Annotations'])

    # Loop through the subject IDs in the original annotations
    for subjectid, subjectid_df in pre:

        print("Subject ID", subjectid)
        print("Subject Annnotations", subjectid_df)

        # Get image information
        image_path, image_name, frame_name = get_image(subjectid_df.iloc[0], image_dir)

        # Get the reduced annotations for the subject ID
        if (post['Subject ID'] == subjectid).any():
            post_subjectid = post_groups.get_group(subjectid)

            # Get new dataframe
            no_singles = remove_single_clusters(subjectid_df)
            metrics_df = create_image_row(no_singles, subjectid, metrics_df, image_path, image_name, post_subjectid)
            # Compare original annotations to reduction
            compare_pre_post(subjectid_df, post_subjectid, image_path, output_dir, image_name, user)

            # Compare the accuracy of original annotations to reduction
            print(no_singles)
            compare_accuracy(no_singles, post_subjectid, image_path, output_dir, image_name)

            # Get the amount of time spent on one image
            time = get_time_duration(subjectid_df.iloc[0])
            total_duration += time
        else:
            continue

    return total_duration, metrics_df

        # Checks if it is over the number of samples
        # count += 1
        # if count > num_samples:
        #     return


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
    plt.figure(figsize=(20, 10))

    # Plot the original annotations on the left
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="RdYlGn")
    plt.title('Pre-aggregation')

    # Color code the boxes based on the distance from reduced annotation
    if not pre.empty:
        color_code = mpl.colormaps['RdYlGn']

        # Plot pre on the left subplot
        for i, r in pre.iterrows():

            # Extract the values of this annotation
            x, y, w, h, d, iou, label = r[['x', 'y', 'w', 'h', "distance", "iou", "correct_label"]]

            # Color code the IoU or "accuracy"
            color = color_code(iou)

            # Create the figure
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
            plt.gca().add_patch(rect)

            if label == 'Y':
                label_color = 'green'
            else:
                label_color = 'red'

            # Plot the class label on the bbox
            plt.text(x + w * 0.02,
                    y + h * 0.98,
                    r['label'],
                    color='white', fontsize=8,
                    ha='left', va='top',
                    bbox=dict(facecolor=label_color, alpha=0.5))

        # Add the accuracy colorbar
        cbar = plt.colorbar(cmap='RdYlGn', location="bottom")
        cbar.set_label("Bounding Box Accuracy")


    # Plot reduced on the right subplot
    plt.subplot(1, 2, 2)
    plt.imshow(image, cmap="RdYlGn")
    for i, r in post.iterrows():

        # Extract the values of this annotation
        x, y, w, h = r[['x', 'y', 'w', 'h']]
        y = float(y)

        # Create the figure
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='green', facecolor='none')
        plt.gca().add_patch(rect)

        # Plot the class label on the bbox
        plt.text(x + w * 0.02,
                y + h * 0.98,
                r['label'],
                color='white', fontsize=8,
                ha='left', va='top',
                bbox=dict(facecolor='green', alpha=0.5))
    plt.title('Post aggregation')
    # Add legend outside the plot
    red_patch = patches.Patch(color='red', label='Incorrect')
    green_patch = patches.Patch(color='green', label='Correct')
    plt.legend(handles=[red_patch, green_patch], title="Label Classification Accuracy", loc='lower center',
               bbox_to_anchor=(1, 1))

    # Save the figure to the output directory
    plt.savefig(f"{output_dir}\\Accuracy\\{image_name}", bbox_inches='tight')
    plt.close()


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
    plt.figure(figsize=(20, 10))

    # Plot pre on the left subplot
    plt.subplot(1, 2, 1)
    plt.imshow(image)
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
        plt.gca().add_patch(rect)

        # Plot the class label on the bbox
        plt.text(x + w * 0.02,
                y + h * 0.98,
                r['label'],
                color='white', fontsize=8,
                ha='left', va='top',
                bbox=dict(facecolor=edge_color, alpha=0.5))
    plt.title("Original annotations")

    # Plot image 2 on the right subplot
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    for i, r in post.iterrows():
        # Extract the values of this annotation
        x, y, w, h = r[['x', 'y', 'w', 'h']]
        y = float(y)

        # Create the figure
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='green', facecolor='none')
        plt.gca().add_patch(rect)

        # Plot the class label on the bbox
        plt.text(x + w * 0.02,
                y + h * 0.98,
                r['label'],
                color='white', fontsize=8,
                ha='left', va='top',
                bbox=dict(facecolor='green', alpha=0.5))
    plt.title('Reduced annotations')

    # Save the figure
    plt.savefig(f"{output_dir}\\User_vs_Reduction\\{image_name}", bbox_inches='tight')
    plt.close()


#TODO: This function needs to be edited + workshopped
def user_average(df, reduced_df):
    """
    This function creates a new dataframe with the average IoU and number of annotations for each user.

    Args:
        df (Pandas dataframe): The original dataframe
    """

    # Remove single clusters
    df = remove_single_clusters(df)

    users = df.groupby(['user_name'])

    # Initialize empty dataframe
    df = pd.DataFrame(columns=['User', 'Average_IoU', 'Number_of_Annotations', 'Average_Precision'])

    # Loop through users
    for user, user_df in users:

        # Find the average IoU and username
        average_iou = user_df['iou'].mean()
        user_name = user_df.iloc[0]['user_name']

        ap = average_precision(user_df, reduced_df)

        # Add to dataframe
        new_row = {'User': user_name, 'Average_IoU': average_iou,
                   'Number_of_Annotations': len(user_df), 'Average_Precision': ap}
        df = df._append(new_row, ignore_index=True)

    return df


def average_precision(user_df, reduced_df):

    precision_array = [0.0]
    recall_array = [0.0]

    # Make the subject ID's integers
    user_df['Subject ID'] = user_df['Subject ID'].astype(int)
    reduced_df['Subject ID'] = reduced_df['Subject ID'].astype(int)

    # Group both dataframes by subject ID
    user_df = user_df.groupby("Subject ID")
    reduced_groups = reduced_df.groupby("Subject ID")

    for subjectid, subjectid_df in user_df:

        # Get the reduced annotations for the subject ID
        if (reduced_df['Subject ID'] == subjectid).any():
            reduced_subjectid = reduced_groups.get_group(subjectid)

            # Get the precision
            tp = subjectid_df['correct_label'].eq('Y').sum()
            total = len(subjectid_df)
            if total > 0:
                precision = tp / total
            else:
                precision = 0

            # Get the recall
            if len(reduced_subjectid) > total:
                fn = len(reduced_subjectid) - total
            else:
                fn = 0
            recall = tp / (tp + fn)

            # Add to arrays
            precision_array.append(precision)
            recall_array.append(recall)
        else:
            continue

    ap_df = pd.DataFrame({'Precision': precision_array, 'Recall': recall_array})
    ap_df.sort_values(by='Recall', inplace=True)
    ap = auc(ap_df['Recall'], ap_df['Precision'])

    return ap






def plot_user_info(df, output_dir):
    """
    This function plots the information for all users looking at user IoU and the number of annotations for each user.

    Args:
        df (Pandas dataframe): The original dataframe with all user annotations
        output_dir (str): The path to the output directory where the plot should be saved
    """

    # Subset and sort the dataframe
    subset_df = df[df['Number_of_Annotations'] > 10]
    subset_df.sort_values(by='Average_IoU', ascending=False, inplace=True)
    subset_df = subset_df.head(100)

    # First plot for IoU against user
    # Plot the user average
    plt.figure(figsize=(20, 10))
    plt.bar(subset_df['User'], subset_df['Average_IoU'])

    # Set axis labels
    plt.xlabel("Users")
    plt.ylabel("Average IoU")
    plt.xticks(rotation=90, fontsize=5)

    # Save the plot
    plt.title('Users Average IoU With Reduced Box')
    plt.savefig(f"{output_dir}\\user_iou.jpg", bbox_inches='tight')
    plt.close()


    # Second plot for IoU against number of annotations
    # Plot the average distance by number of annotations
    plt.figure(figsize=(20, 10))
    plt.scatter(subset_df['Number_of_Annotations'], subset_df['Average_IoU'])

    # Set axis labels
    plt.ylabel("Average IoU")
    plt.xlabel("Number of Annotations")
    plt.title("Relationship Between # of Annotations and IoU with Reduced Box")

    # Create regression line
    model = LinearRegression()
    model.fit(subset_df[['Number_of_Annotations']], subset_df['Average_IoU'])
    line = model.predict(subset_df[['Number_of_Annotations']])

    # Plot line of best fit
    plt.plot(subset_df['Number_of_Annotations'], line, color='red')

    # Save the plot
    plt.savefig(f"{output_dir}\\annotations_vs_user_iou.jpg", bbox_inches='tight')
    plt.close()


def plot_image_info(df, output_dir):
    """
    This function plots the information for all users looking at user IoU and the number of annotations for each user.

    Args:
        df (Pandas dataframe): The original dataframe with all user annotations
        output_dir (str): The path to the output directory where the plot should be saved
    """

    print("plot image", df)
    # Subset and sort the dataframe
   # subset_df = df[df['Number_of_Annotations'] > 10]
    subset_df = df
    subset_df.sort_values(by='Average_IoU', ascending=False, inplace=True)
    print("test")
   #  subset_df = subset_df.head(100)

    # Second plot for IoU against number of annotations
    # Plot the average distance by number of annotations
    plt.figure(figsize=(20, 10))
    plt.scatter(subset_df['Number_of_Annotations'], subset_df['Average_IoU'])

    # Set axis labels
    plt.ylabel("Average IoU")
    plt.xlabel("Number of Annotations")
    plt.title("Relationship Between # of Annotations and IoU with Reduced Box")

    # Create regression line
    model = LinearRegression()
    model.fit(subset_df[['Number_of_Annotations']], subset_df['Average_IoU'])
    line = model.predict(subset_df[['Number_of_Annotations']])

    # Plot line of best fit
    plt.plot(subset_df['Number_of_Annotations'], line, color='red')

    # Save the plot
    plt.savefig(f"{output_dir}\\annotations_vs_image_iou.jpg", bbox_inches='tight')
    plt.close()


def user_information(reduced, original, user, image_dir, output_dir, user_df):
    """
    This function looks at providing information regarding a specific user. It plots the same visualizations only for
    the user and creates a text file with information regarding the users status and ability.

    Args:
        reduced (Pandas dataframe): The reduced annotation dataframe
        original (Pandas dataframe): The original user annotation dataframe
        user (str): The user's name
        image_dir (str): The filepath to the image directory
        output_dir (str): The filepath to the output directory
        user_df (Pandas dataframe): Dataframe containing information on all the users
    """

    # Sorts user dataframe
    user_df.sort_values(by='Average_Precision', ascending=False, inplace=True)
    user_df.reset_index(drop=True, inplace=True)

    # Set output directory for the user
    output_dir = f"{output_dir}\\{user}"

    # Make directory for the user
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}\\User_vs_Reduction", exist_ok=True)
    os.makedirs(f"{output_dir}\\Accuracy", exist_ok=True)

    # Finds the specific user
    user_info = user_df.loc[user_df['User'] == user]

    # Gets the users ranking
    ranking = user_info.index.tolist()

    ap = user_info['Average_Precision']

    # Find all the users annotations
    user_subset = original[original['user_name'] == user]

    # Find the subject IDS specific for the user
    subjectids = user_subset['Subject ID'].unique().tolist()
    number_of_subjectids = len(subjectids)

    # Find the reduced annotations that correspond to the users
    reduced_subset = reduced[reduced['Subject ID'].isin(subjectids)]

    # Run through the images for the user
    total_duration, images_df = group_annotations(user_subset, reduced_subset, image_dir,
                                                  number_of_subjectids, output_dir, True)

    print("images df", images_df)
    images_df.sort_values(by='Recall', ascending=False, inplace=True)

    # Finds the users most best and worst images (OPTIONAL)
    find_difficult_images(images_df, output_dir)

    # Get rid of the single clusters
    user_subset = remove_single_clusters(user_subset)

    # Get the average accuracy of bbox and classification
    average_bbox_accuracy = user_subset['iou'].mean()

    # Calculate the proportion the user is correctly classifying species
    yes_count = user_subset['correct_label'].eq('Y').sum()
    total = len(user_subset)
    if total > 0:
        proportion = yes_count / total
    else:
        proportion = 0

    # Create text file for user
    txt_file = open(f"{output_dir}\\{user}.txt", "w")

    txt_file.write(f"Username: {user}\n")
    txt_file.write(f"Time spent annotating: {total_duration}\n")
    txt_file.write(f"Total number of annotations: {len(user_subset)}\n")
    txt_file.write(f"Total number of images annotated: {number_of_subjectids}\n")
    txt_file.write(f"IoU accuracy: {average_bbox_accuracy * 100}\n")
    txt_file.write(f"Identification accuracy: {proportion * 100}\n")
    txt_file.write(f"Average Precision: {ap * 100}")
    txt_file.write(f"Your ranking (based on Average Precision): {ranking[0]}")

    txt_file.close()


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
    plt.title(f"IoU Distribution of Bounding Boxes for {user}")

    # Add legend outside the plot
    red_patch = patches.Patch(color='red', label='No')
    green_patch = patches.Patch(color='green', label='Yes')
    plt.legend(handles=[red_patch, green_patch], title="Correct Classification", loc='upper left', bbox_to_anchor=(1, 1))

    # Save the plot
    plt.savefig(f"{output_dir}\\distance_distribution.jpg", bbox_inches='tight')
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


def create_image_row(df, subject_id, images_df, image_path, image_name, reduced_df):
    """
    This function creates a new dataframe row with metrics on a specific image/Subject ID.

    Args:
        df (Pandas dataframe): A dataframe of annotations for a specific subject ID
        subject_id (int): The subject ID of an image
        images_df (Pandas dataframe): A dataframe that contains information on metrics for all the images
        image_path (str): Filepath to the image
        image_name (str): What the image should be called
        reduced_df (Pandas dataframe): A dataframe containing the reduced annotations for a subject ID

    Returns:
        images_df (Pandas dataframe): The image dataframe with an additional row containing information on the image
        provided to this function
    """

    # Get the average IoU
    average_iou = df['iou'].mean()

    # Get the precision
    tp = df['correct_label'].eq('Y').sum()
    total = len(df)
    if total > 0:
        precision = tp / total
    else:
        precision = 0

    # Get the recall
    if len(reduced_df) > total:
        fn = len(reduced_df) - total
    else:
        fn = 0
    recall = tp / (tp + fn)

    # Create new row
    new_row = {'Average_IoU': average_iou, 'Precision': precision, 'Recall': recall, 'Subject ID': subject_id,
               'image_path': image_path, 'image_name': image_name, 'Number_of_Annotations': total}

    # Add a new row to
    images_df = images_df._append(new_row, ignore_index=True)

    return images_df


def find_difficult_images(df, output_dir):
    """
    This function finds the most and least difficult images for a dataframe containing information
    on a number of images.

    Args:
        df (Pandas dataframe): An image dataframe in the format provided from create_image_row
        output_dir (str): Path to the output directory
    """

    output_dir = f"{output_dir}\\Image Difficulty"

    # Make all the directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}\\Easy\\BBOX", exist_ok=True)
    os.makedirs(f"{output_dir}\\Hard\\BBOX", exist_ok=True)
    os.makedirs(f"{output_dir}\\Easy\\Classification", exist_ok=True)
    os.makedirs(f"{output_dir}\\Hard\\Classification", exist_ok=True)


    # Create text file for images
    txt_file = open(f"{output_dir}\\hard_and_easy_images.txt", "w")

    # Sort the dataframe based on IoU
    df = df.sort_values(by='Average_IoU', ascending=False)

    # Add the top 10 easiest images to make bounding boxes on
    txt_file.write(f"The top 10 easiest images for users to draw bounding boxes were:\n")
    for i, row in df.head(10).iterrows():
        txt_file.write(f"{row['image_name']}\n")
        shutil.copy(row['image_path'], f"{output_dir}\\Easy\\BBOX")

    # Add the top 10 hardest images to make bounding boxes on
    txt_file.write(f"The top 10 hardest images for users to draw bounding boxes were:\n")
    for i, row in df.tail(10).iterrows():
        txt_file.write(f"{row['image_name']}\n")
        shutil.copy(row['image_path'], f"{output_dir}\\Hard\\BBOX")


    # Sort the dataframe by class accuracy
    df = df.sort_values(by='Precision', ascending=False)

    txt_file.write("\n")

    # Add the top 10 easiest images to correctly classify
    txt_file.write(f"The top 10 easiest images for users to correctly classify species were:\n")
    for i, row in df.head(10).iterrows():
        txt_file.write(f"{row['image_name']}\n")
        shutil.copy(row['image_path'], f"{output_dir}\\Easy\\Classification")

    # Add the top 10 hardest images to correctly classify
    txt_file.write(f"The top 10 hardest images for users to correctly classify species were:\n")
    for i, row in df.tail(10).iterrows():
        txt_file.write(f"{row['image_name']}\n")
        shutil.copy(row['image_path'], f"{output_dir}\\Hard\\Classification")

    txt_file.close()

# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize non-reduced and reduced annotations")

    parser.add_argument("--reduced_csv", type=str,
                        help="The CSV of reduced annotations")

    parser.add_argument("--full_csv", type=str,
                        help="The CSV of all annotations")

    parser.add_argument("--image_dir", type=str,
                        default='./Data',
                        help="The image directory")

    parser.add_argument("--output_dir", type=str,
                        default="./Test",
                        help="Output directory")

    # TODO: This will eventually be removed
    parser.add_argument("--num_samples", type=int,
                        default=1,
                        help="Number of samples to run through")

    parser.add_argument("--user", type=str, nargs='*',
                        default=None,
                        help="A list of usernames")

    parser.add_argument("--num_users", type=int,
                        default=2,
                        help="The number of random users to sample")


    args = parser.parse_args()

    # Parse out arguments
    reduced_csv = args.reduced_csv
    full_csv = args.full_csv
    num_samples = args.num_samples
    user_names = args.user
    image_dir = args.image_dir
    num_users = args.num_users

    # Turn both csv files into pandas dataframes
    pre = pd.read_csv(full_csv)
    post = pd.read_csv(reduced_csv)

    # Create output directories
    output_dir = f"{args.output_dir}\\Visualizations"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}\\User_vs_Reduction", exist_ok=True)
    os.makedirs(f"{output_dir}\\Accuracy", exist_ok=True)
    os.makedirs(f"{output_dir}\\Users", exist_ok=True)


    try:

        # Creates a dataframe for all users
        user_df = user_average(pre, post)
        print("user df", user_df)

        if user_names is None and num_users is None:
            total_duration, images_df = group_annotations(pre, post, image_dir, num_samples, output_dir, False)

            #find_difficult_images(images_df, output_dir)

            #print("user df", user_df)
            print("images df", images_df)

            plot_user_info(user_df, output_dir)
            plot_image_info(images_df, output_dir)

        else:
            output_dir = f"{output_dir}\\Users"

            if user_names is not None:

                for user_name in user_names:
                    user_information(post, pre, user_name, image_dir, output_dir, user_df)

            else:
                usernames = pre['user_name'].unique().tolist()

                # Pull a random row
                user_names = random.sample(usernames, num_users)

                for user_name in user_names:
                    print(user_name)
                    user_information(post, pre, user_name, image_dir, output_dir, user_df)


        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
