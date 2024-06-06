import argparse
import sys
import glob

import matplotlib
import pandas as pd
import statistics

from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

import cv2
import os
import numpy as np
import supervision as sv
from reduce_annotations import get_image


def group_annotations(pre, post, image_dir, num_samples, output_dir):

    # Turn both csv files into pandas dataframes
    pre = pd.read_csv(pre)
    post = pd.read_csv(post)

    # Make the subject ID's integers
    pre['Subject ID'] = pre['Subject ID'].astype(int)
    post['Subject ID'] = post['Subject ID'].astype(int)

    # Group both dataframes by subject ID
    pre = pre.groupby("Subject ID")
    post = post.groupby("Subject ID")

    count = 0

    for subjectid, subjectid_df in pre:

        # get image information
        image_path, image_name, frame_name = get_image(subjectid_df.iloc[0], image_dir)

        post_subjectid = post.get_group(subjectid)

        print("pre", subjectid_df, subjectid)
        print("post", post_subjectid)

        # Compare original annotations to reduction
        compare_pre_post(subjectid_df, post_subjectid, image_path, output_dir, image_name)

        compare_accuracy(subjectid_df, post_subjectid, image_path, output_dir, image_name)
        # Checks if it is over the number of samples
        count += 1
        if count > num_samples:
            return


def compare_accuracy(pre, post, image_path, output_dir, image_name):

    # Get a color mapping for all the distances first
    # distances = pre['distance'].unique().tolist()
    # color_codes = {username: tuple(np.random.rand(3, )) for username in usernames}

    color_code = mpl.colormaps['RdYlGn_r']
    pre['distance'].fillna(100, inplace=True)

    image = plt.imread(image_path)

    # Plot the images side by side
    plt.figure(figsize=(20, 10))

    # Plot pre on the left subplot
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    for i, r in pre.iterrows():
        # Extract the values of this annotation
        x, y, w, h, d = r[['x', 'y', 'w', 'h', "distance"]]
        d = d / 100
        color = color_code(d)

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
    plt.title('Pre-aggregation')

    # Plot image 2 on the right subplot
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    for i, r in post.iterrows():
        # Extract the values of this annotation
        x, y, w, h = r[['x', 'y', 'w', 'h']]
        # Create the figure
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='black', facecolor='none')
        plt.gca().add_patch(rect)

        # Plot the class label on the bbox
        plt.text(x + w * 0.02,
                y + h * 0.98,
                r['label'],
                color='white', fontsize=8,
                ha='left', va='top',
                bbox=dict(facecolor='black', alpha=0.5))
    plt.title('Post aggregation')

    #plt.imshow(image)
    #plt.show()

    plt.savefig(f"{output_dir}\\Accuracy\\{image_name}", bbox_inches='tight')

def compare_pre_post(pre, post, image_path, output_dir, image_name):

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


    args = parser.parse_args()

    # parse out arguments
    reduced_csv = args.reduced_csv
    full_csv = args.full_csv
    num_samples = args.num_samples

    image_dir = args.image_dir
    output_dir = f"{args.output_dir}\\Visualizations"

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}\\User_vs_Reduction", exist_ok=True)
    os.makedirs(f"{output_dir}\\Accuracy", exist_ok=True)


    try:
        group_annotations(full_csv, reduced_csv, image_dir, num_samples, output_dir)

        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()