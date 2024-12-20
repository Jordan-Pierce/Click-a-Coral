import os
import glob
import json
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------

def clean_csv_file_legacy(input_csv, label_dir):
    """
    This is a function to clean the CSV file provided from Zooniverse
    containing user annotations. The workflow (ID 25828) was changed in
    04/2024, and this function will not work with the newest workflow.

    We had some user annotations (~9k?) in the old workflow that could still be used...

    Use this function as an example for how to clean data for the newest
    workflow. Use both functions to extract and clean the data for the
    respective workflows, and then combine them into a single pandas dataframe.
    """
    # Output CSV for creating training data
    output_csv = f"{label_dir}/extracted_data_legacy.csv"

    # Read the file
    df = pd.read_csv(input_csv)
    # Filter by workflow id and version
    df = df[df['workflow_id'] == 25828]
    df = df[df['workflow_version'] == 355.143]

    # Change the dataframe in place
    #unjson_dataframe(df)

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


def clean_csv_file(input_csv, label_dir,  workflow_id, version):
    """
    This is a function to clean the CSV file provided from Zooniverse
    containing user annotations.

    Args:
        input_csv (str): The original csv file from Click-a-Coral
        label_dir (str): The output directory
        workflow_id (int): The current workflow id
        version (int): The current workflow version

    Returns:
        clean_df (Pandas dataframe): The cleaned dataframe
        output_csv (str): The cleaned dataframe as a csv
    """
    # Output CSV for creating training data
    output_csv = f"{label_dir}/extracted_data.csv"

    # Read the file
    df = pd.read_csv(input_csv, low_memory=False)
    # Filter by workflow id and version
    df = df[df['workflow_id'] == workflow_id]
    df = df[df['workflow_version'] == version]

    # Loop through the dataframe annotations, and restructure it
    clean_df = []

    # Truncate counter
    truncate_counter = 0
    point_counter = 0

    for i, r in tqdm(df.iterrows(), total=len(df), desc="Cleaning Data"):

        # Get the image metadata
        subject_data = json.loads(r['subject_data'])
        subject_id = next(iter(subject_data))
        subject_data = subject_data[subject_id]
        subject_data['Subject ID'] = subject_id

        # Get the image start and end time
        meta_data = json.loads(r['metadata'])
        start_time = meta_data['started_at']
        end_time = meta_data['finished_at']

        # Add to subject_data
        subject_data['started_at'] = start_time
        subject_data['finished_at'] = end_time

        # Convert from string to list of dicts
        # Get all but the last one, since last is NULL
        annotations = json.loads(r['annotations'])[:-1]

        for i in range(0, len(annotations), 3):

            try:
                # Access three elements at a time
                t2, t1, t0 = annotations[i: i + 3]

                # Loop through individual annotations in t0
                for bbox in t0['value']:

                    x = bbox['x']
                    y = bbox['y']
                    w = bbox['width']
                    h = bbox['height']

                    image_width = subject_data['Width']
                    image_height = subject_data['Height']

                    # Checks if right corners of bounding box are outside the image width
                    if x + w > image_width:
                        w = (image_width - x)
                        truncate_counter += 1

                    # Checks if bottom corners of bounding box are outside the image width
                    if y + h > image_height:
                        h = (image_height - y)
                        truncate_counter += 1

                    # Checks if left corners of bounding box are outside the image width
                    if x < 0:
                        x = 0
                        truncate_counter += 1

                    # Checks if top corners of bounding box are outside the image width
                    if y < 0:
                        y = 0
                        truncate_counter += 1

                    # Checks if the box is just a singular point or line
                    if w == 0 or h == 0:
                        point_counter += 1
                        continue

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
                print(f"{e} \n There is something wrong with this row.")

    # Create a pandas dataframe, save it
    clean_df = pd.DataFrame(clean_df)
    clean_df.to_csv(output_csv)

    return clean_df, output_csv


def plot_samples(df, image_dir, output_dir, num_samples, include_legend):
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

                # Plot the class label on the bbox
                ax.text(x + w * 0.02,
                        y + h * 0.98,
                        r['label'],
                        color='white', fontsize=8,
                        ha='left', va='top',
                        bbox=dict(facecolor=edge_color, alpha=0.5))

            # Add the user's color and number of annotations to legend
            legend_labels.append(f'User {user} - {num_annotations} boxes')
            edge_patch = patches.Patch(color=edge_color, label=f'User {u_idx + 1}')
            edge_patches.append(edge_patch)

        # Add legend outside the plot
        if include_legend:
            # Add legend outside the plot
            ax.legend(handles=edge_patches, labels=legend_labels, loc='upper left', bbox_to_anchor=(1, 1))

        # Save with same name as frame in examples folder
        plt.title(f"Media {media_id} - Frame {frame_name}")
        plt.suptitle(f"{len(subset)} Annotations")
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

    parser.add_argument("--workflow_id", type=int,
                        default=26428,
                        help="Workflow ID.")

    parser.add_argument("--version", type=float,
                        default=48.28,
                        help="Version.")

    parser.add_argument("--input_csv", type=str,
                        help="Path to the input CSV file.")

    parser.add_argument("--image_dir", type=str,
                        default="./Data",
                        help="Path to the image directory.")

    parser.add_argument("--label_dir", type=str,
                        default="./Extracted",
                        help="Path to the label directory.")

    parser.add_argument("--num_samples", type=int,
                        default=100,
                        help="Number of samples to plot.")

    parser.add_argument("--legend_flag", action="store_true",
                        help="Include if user legend should be on the plot")

    args = parser.parse_args()

    # Extract args
    workflow_id = args.workflow_id
    version = args.version
    num_samples = args.num_samples
    include_legend = args.legend_flag

    # Extract the shapes for the workflow
    input_csv = args.input_csv
    image_dir = f"{args.image_dir}"
    label_dir = f"{args.label_dir}"
    sample_dir = f"{label_dir}\\user_samples"

    assert os.path.exists(input_csv), "ERROR: Input CSV provided does not exist"
    assert os.path.exists(image_dir), "ERROR: Image directory provided does not exist"

    # Make the label directory
    os.makedirs(sample_dir, exist_ok=True)

    print("Workflow ID:", workflow_id)
    print("Version:", version)
    print("Input CSV:", input_csv)
    print("Image Directory:", image_dir)
    print("Label Directory:", label_dir)
    print("Include Legend:", include_legend)

    # Clean the classification csv, convert to a dataframe for creating training data
    df, path = clean_csv_file(input_csv, label_dir, workflow_id, version)

    # Plot some samples
    plot_samples(df, image_dir, sample_dir, num_samples, include_legend)


if __name__ == "__main__":
    main()