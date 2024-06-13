import os
import argparse

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
         num_samples (int): The number of images/subject IDs to run through
         output_dir (str): The filepath to the output directory
         user (bool): Whether a user has been provided in the arguments
    """

    # Make the subject ID's integers
    pre['Subject ID'] = pre['Subject ID'].astype(int)
    post['Subject ID'] = post['Subject ID'].astype(int)

    # Group both dataframes by subject ID
    pre = pre.groupby("Subject ID")
    post = post.groupby("Subject ID")

    count = 0

    # Loop through the subject IDs in the original annotations
    for subjectid, subjectid_df in pre:

        # Get image information
        image_path, image_name, frame_name = get_image(subjectid_df.iloc[0], image_dir)

        # Get the reduced annotations for the subject ID
        post_subjectid = post.get_group(subjectid)

        print("pre", subjectid_df, subjectid)
        print("post", post_subjectid)

        # Compare original annotations to reduction
        compare_pre_post(subjectid_df, post_subjectid, image_path, output_dir, image_name, user)

        # Compare the accuracy of original annotations to reduction
        compare_accuracy(subjectid_df, post_subjectid, image_path, output_dir, image_name)

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

    # Removes single clusters
    pre = remove_single_clusters(pre)

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
        pre['distance'].fillna(100, inplace=True)

        # Plot pre on the left subplot
        for i, r in pre.iterrows():

            # Extract the values of this annotation
            x, y, w, h, d, iou = r[['x', 'y', 'w', 'h', "distance", "iou"]]

            # Get the distance or "accuracy"
            d = d / 100
            color = color_code(iou)

            # Create the figure
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
            plt.gca().add_patch(rect)

            # Plot the class label on the bbox
            plt.text(x + w * 0.02,
                    y + h * 0.98,
                    r['label'],
                    color='white', fontsize=8,
                    ha='left', va='top',
                    bbox=dict(facecolor='black', alpha=0.5))

        # Add the accuracy colorbar
        cbar = plt.colorbar(cmap='RdYlGn', location="bottom")
        cbar.set_label("Accuracy")


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

    # Save the figure to the output directory
    plt.savefig(f"{output_dir}\\Accuracy\\{image_name}", bbox_inches='tight')

def compare_pre_post(pre, post, image_path, output_dir, image_name, user):

    # Get a color mapping for all the users first
    usernames = pre['user_name'].unique().tolist()
    color_codes = {username: tuple(np.random.rand(3, )) for username in usernames}

    image = plt.imread(image_path)

    # Plot the images side by side
    plt.figure(figsize=(20, 10))

    # Plot pre on the left subplot
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    for i, r in pre.iterrows():
        # Extract the values of this annotation
        x, y, w, h = r[['x', 'y', 'w', 'h']]

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

    #plt.imshow(image)
    #plt.show()

    plt.savefig(f"{output_dir}\\User_vs_Reduction\\{image_name}", bbox_inches='tight')

def user_average(df, output_dir):

    # Remove single clusters (OPTIONAL)
    df = remove_single_clusters(df)
    print(df)

    users = df.groupby(['user_name'])

    # Initialize empty dataframe
    df = pd.DataFrame(columns=['User', 'Average_Distance', 'Number_of_Annotations'])

    # Loop through users
    for user, user_df in users:

        average_distance = user_df['distance'].mean()
        user_name = user_df.iloc[0]['user_name']

        # Add to dataframe
        new_row = {'User': user_name, 'Average_Distance': average_distance, 'Number_of_Annotations': len(user_df)}
        df = df._append(new_row, ignore_index=True)

    # Subset and sort the dataframe
    subset_df = df[df['Number_of_Annotations'] > 10]
    subset_df = subset_df.sort_values(by='Average_Distance')

    # Plot the user average
    plt.figure(figsize=(20, 10))
    plt.bar(subset_df['User'], subset_df['Average_Distance'])

    # Set axis labels
    plt.xlabel("Users")
    plt.ylabel("Average Distance from 'Best Fit'")
    plt.xticks(rotation=90, fontsize=5)

    # Save the plot
    plt.title('Users Average Distance From Reduced Box')
    plt.savefig(f"{output_dir}\\user_distance_no_single.jpg", bbox_inches='tight')


    # Plot the average distance by number of annotations
    plt.figure(figsize=(20, 10))
    plt.scatter(subset_df['Number_of_Annotations'], subset_df['Average_Distance'])

    # Set axis labels
    plt.ylabel("Average Distance from Reduced Box")
    plt.xlabel("Number of Annotations")
    plt.title("Relationship Between # of Annotations and Distance from Reduction")

    # Create regression line
    model = LinearRegression()
    model.fit(subset_df[['Number_of_Annotations']], subset_df['Average_Distance'])
    line = model.predict(subset_df[['Number_of_Annotations']])

    # Plot line of best fit
    plt.plot(subset_df['Number_of_Annotations'], line, color='red')

    # Save the plot
    plt.savefig(f"{output_dir}\\annotations_vs_distance_no_single.jpg", bbox_inches='tight')


def user_information(reduced, original, user, image_dir, output_dir):

    # Remove single clusters from the reduced annotations
    reduced = remove_single_clusters(reduced)

    # Find all the users annotations
    user_subset = original[original['user_name'] == user]

    # Find the subject IDS specific for the user
    subjectids = user_subset['Subject ID'].unique().tolist()
    number_of_subjectids = len(subjectids)

    # Find the reduced annotations that correspond to the users
    reduced_subset = reduced[reduced['Subject ID'].isin(subjectids)]

    #TODO: This "gets" how much time the user spent annotating but either need
    # a better way to do this, or not do it at all

    # subset['created_at'] = pd.to_datetime(subset['created_at']).dt.tz_convert(None)
    #
    # print(subset)
    # dates = subset.groupby(subset['created_at'].dt.date)
    #
    # total_duration = pd.Timedelta(0, unit='s')
    # annotation_number = len(subset)
    #
    # for date, date_df in dates:
    #     print(date, date_df)
    #
    #     times = date_df['created_at'].unique().tolist()
    #     times = sorted(times)
    #     print(times)
    #
    #     # Get duration
    #     if len(times) > 1:
    #         duration = times[-1] - times[0]
    #         print(duration)
    #
    #         total_duration += duration
    #
    # print(total_duration)

    # Set output directory for the user
    output_dir = f"{output_dir}\\{user}"

    # Make directory for the user
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}\\User_vs_Reduction", exist_ok=True)
    os.makedirs(f"{output_dir}\\Accuracy", exist_ok=True)

    # Run through the images for the user
    group_annotations(user_subset, reduced_subset, image_dir, number_of_subjectids, output_dir, True)

    # Get rid of the single clusters
    user_subset = remove_single_clusters(user_subset)


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


def main():
    parser = argparse.ArgumentParser(description="Visualize non-reduced and reduced annotations")

    parser.add_argument("-reduced_csv", type=str,
                        help="The CSV of reduced annotations")

    parser.add_argument("-full_csv", type=str,
                        help="The CSV of all annotations")

    parser.add_argument("-image_dir", type=str,
                        help="The image directory")

    parser.add_argument("-output_dir", type=str,
                         help="Output directory")

    parser.add_argument("-num_samples", type=int,
                        default=1,
                        help="Number of samples to run through")

    parser.add_argument("-user", type=str, nargs='*',
                        default=None,
                        help="A list of usernames")


    args = parser.parse_args()

    # parse out arguments
    reduced_csv = args.reduced_csv
    full_csv = args.full_csv
    num_samples = args.num_samples
    user_names = args.user

    image_dir = args.image_dir
    output_dir = f"{args.output_dir}\\Visualizations"

    # Turn both csv files into pandas dataframes
    pre = pd.read_csv(full_csv)
    post = pd.read_csv(reduced_csv)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}\\User_vs_Reduction", exist_ok=True)
    os.makedirs(f"{output_dir}\\Accuracy", exist_ok=True)


    try:

        if user_names is None:
            group_annotations(pre, post, image_dir, num_samples, output_dir, False)

            user_average(pre, output_dir)
        else:
            for user_name in user_names:
                user_information(post, pre, user_name, image_dir, output_dir)

        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()