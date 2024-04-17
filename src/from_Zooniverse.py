import os
import glob
import json
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from panoptes_aggregation.csv_utils import unjson_dataframe


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------

def clean_csv_file(input_csv, label_dir, workflow_id, version):
    """

    """
    # Output CSV for creating training data
    output_csv = f"{label_dir}/extracted_data.csv"

    # Read the file
    df = pd.read_csv(input_csv)
    # Filter by workflow id and version
    df = df[df['workflow_id'] >= workflow_id]
    df = df[df['workflow_version'] >= version]

    # Change the dataframe in place
    unjson_dataframe(df)

    # Loop through the dataframe annotations, and restructure it
    clean_df = []

    for i, r in tqdm(df.iterrows()):

        # Get the image metadata
        subject_data = json.loads(r['subject_data'])
        subject_id = next(iter(subject_data))
        subject_data = subject_data[subject_id]
        subject_data['Subject ID'] = subject_id

        # Convert from string to list of dicts
        # Get all but the last one, since last is NULL
        annotations = json.loads(r['annotations'])[:-1]

        for i in range(0, len(annotations), 3):

            try:
                # Access three elements at a time
                t2, t0, t1 = annotations[i: i + 3]

                # Extract the bounding box
                bbox = t0['value'][0]

                x = bbox['x']
                y = bbox['y']
                w = bbox['width']
                h = bbox['height']

                # Extract the label
                choice = t1['value'][0]
                label = choice['choice']

                # Create a subset of the row for this annotation
                new_row = r.copy()[['classification_id', 'user_name', 'user_id', 'user_ip', 'created_at']]

                # Create a dict that contains all information
                new_row = {**new_row, **subject_data}

                new_row['x'] = x
                new_row['y'] = y
                new_row['w'] = w
                new_row['h'] = h
                new_row['label'] = label

                # Add to the updated dataframe
                clean_df.append(new_row)

            except Exception as e:
                # There isn't box coordinates, not sure why this happens sometimes?
                pass

    # Create a pandas dataframe, save it
    clean_df = pd.DataFrame(clean_df)
    clean_df.to_csv(output_csv)

    return clean_df, output_csv


def plot_samples(df, image_dir, output_dir, num_samples):
    """

    """
    # Get a color mapping for all the users first
    usernames = df['user_name'].unique().tolist()
    color_codes = {username: tuple(np.random.rand(3, )) for username in usernames}

    for _ in range(num_samples):

        # Pull a random row
        r = df.sample(n=1)

        # Get the meta
        media_id = r['Media ID'].values[0]
        frame_name = r['Frame Name'].values[0]

        # Get media folder, the frame path
        media_folders = glob.glob(f"{image_dir}\\*")
        media_folder = [f for f in media_folders if str(media_id) in f][0]
        frame_path = f"{media_folder}\\frames\\{frame_name}"

        # Make sure the file exists before opening
        if not os.path.exists(frame_path):
            print(f"ERROR: Could not find {frame_name} in {media_folder} folder")
            continue

        # Set the output file name
        output_file = f"{output_dir}\\{frame_name}"

        # Skip if it already exists
        if os.path.exists(output_file):
            continue

        # Open the image
        image = plt.imread(frame_path)

        # Get all the boxes for this image
        subset = df[(df['Media ID'] == media_id) & (df['Frame Name'] == frame_name)]
        users = subset['user_name'].unique().tolist()

        # Create the figure
        fig, ax = plt.subplots(figsize=(20, 10))

        # Lists to store information for the legend
        legend_labels = []
        edge_patches = []
        boxes_per_frame = []

        for u_idx, user in enumerate(users):

            # Get the user's color
            edge_color = color_codes[user]

            # Get the user's annotations
            annotations = subset[subset['user_name'] == user]
            num_annotations = len(annotations)

            # Loop through all the user's annotations
            for i, r in annotations.iterrows():

                # Extract the values of this annotation
                x, y, w, h = r[['x', 'y', 'w', 'h']]

                # Plot the annotation the user made
                rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=edge_color, facecolor='none')
                ax.add_patch(rect)

            # Add the user's color and number of annotations to legend
            legend_labels.append(f'User {user} - {num_annotations} boxes')
            edge_patch = patches.Patch(color=edge_color, label=f'User {u_idx + 1}')
            edge_patches.append(edge_patch)

        # Add legend outside the plot
        ax.legend(handles=edge_patches, labels=legend_labels, loc='upper left', bbox_to_anchor=(1, 1))

        # Save with same name as frame in examples folder
        plt.title(f"Media {media_id} - Frame {frame_name}")
        plt.suptitle(f"{len(subset)} annotations")
        plt.imshow(image)
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

def main():
    """

    """
    parser = argparse.ArgumentParser(description="Convert Zooniverse Annotations.")

    parser.add_argument("--season_num", type=int,
                        default=1,
                        help="Season number.")

    parser.add_argument("--workflow_id", type=int,
                        default=25828,
                        help="Workflow ID.")

    parser.add_argument("--version", type=float,
                        default=355.143,
                        help="Version.")

    parser.add_argument("--input_csv", type=str,
                        help="Path to the input CSV file.")

    parser.add_argument("--image_dir", type=str,
                        help="Path to the image directory.")

    parser.add_argument("--label_dir", type=str,
                        help="Path to the label directory.")

    parser.add_argument("--num_samples", type=int,
                        default=100,
                        help="Number of samples to plot.")

    args = parser.parse_args()

    season_num = args.season_num
    workflow_id = args.workflow_id
    version = args.version

    num_samples = args.num_samples

    # Extract the shapes for the workflow
    input_csv = args.input_csv
    image_dir = f"{args.image_dir}\\Season_{season_num}"
    label_dir = f"{args.label_dir}\\Season_{season_num}"
    sample_dir = f"{label_dir}\\user_samples"

    assert os.path.exists(input_csv), "ERROR: Input CSV provided does not exist"
    assert os.path.exists(image_dir), "ERROR: Image directory provided does not exist"

    # Make the label directory
    os.makedirs(sample_dir, exist_ok=True)

    print("Season Number:", season_num)
    print("Workflow ID:", workflow_id)
    print("Version:", version)
    print("Input CSV:", input_csv)
    print("Image Directory:", image_dir)
    print("Label Directory:", label_dir)

    # Clean the classification csv, convert to a dataframe for creating training data
    df, path = clean_csv_file(input_csv, label_dir, workflow_id, version)

    if num_samples:
        plot_samples(df, image_dir, sample_dir, num_samples)


if __name__ == "__main__":
    main()