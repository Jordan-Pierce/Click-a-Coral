import os
import glob
import json
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------


class ZooniverseProcessor:
    """
    A class to process Zooniverse annotation data.
    
    This class provides functionality to clean CSV files from Zooniverse containing
    user annotations and optionally plot samples of the annotations.
    """
    
    def __init__(self, input_classifications_csv=None, output_dir=None, workflow_id=26428, version=48.28):
        """
        Initialize a ZooniverseProcessor instance.
        
        Args:
            input_classifications_csv (str): The original csv file from Click-a-Coral
            output_dir (str): The output directory
            workflow_id (int): The current workflow id
            version (float): The current workflow version
        """
        self.input_classifications_csv = input_classifications_csv
        self.output_dir = output_dir
        self.workflow_id = workflow_id
        self.version = version
        self.clean_df = None
        self.output_csv = None
        self.src_image_dir = None
        self.sample_dir = None

    def print_configuration(self, plot_flag=False, include_legend=False):
        """
        Print the current configuration settings.
        
        Args:
            plot_flag (bool): Whether plotting is enabled
            include_legend (bool): Whether legend is included in plots
        """
        print("Workflow ID:", self.workflow_id)
        print("Version:", self.version)
        print("Input CSV:", self.input_classifications_csv)
        print("Output Directory:", self.output_dir)
        if plot_flag and self.src_image_dir:
            print("Image Directory:", self.src_image_dir)
            if include_legend:
                print("Include Legend:", include_legend)

    def set_image_directory(self, src_image_dir):
        """
        Set the source image directory for plotting.
        
        Args:
            src_image_dir (str): Path to the source image directory
        """
        self.src_image_dir = src_image_dir
        # Create a samples directory within the output directory
        self.sample_dir = f"{self.output_dir}/user_samples"
        os.makedirs(self.sample_dir, exist_ok=True)
        
        print("Image Directory:", self.src_image_dir)
        
        # Validate image directory exists
        assert os.path.exists(self.src_image_dir), "ERROR: Image directory provided does not exist"

    def clean_csv_file(self):
        """
        This is a method to clean the CSV file provided from Zooniverse
        containing user annotations.

        Returns:
            clean_df (Pandas dataframe): The cleaned dataframe
            output_csv (str): The cleaned dataframe as a csv
        """
        # Output CSV for creating training data with a timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_csv = f"{self.output_dir}/extracted_data_{timestamp}.csv"

        # Read the file
        df = pd.read_csv(self.input_classifications_csv, low_memory=False)
        # Filter by workflow id and version
        df = df[df['workflow_id'] == self.workflow_id]
        df = df[df['workflow_version'] == self.version]
        
        # Filter out bad usernames
        if True:
            df = df[~df['user_name'].isin(['jparks1522'])]

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
                    print(f"Issue with row {i} usename '{r['user_name']}': {e}; skipping...")

        # Create a pandas dataframe, save it
        self.clean_df = pd.DataFrame(clean_df)
        self.clean_df.to_csv(self.output_csv)

        return self.clean_df, self.output_csv

    def plot_samples(self, src_image_dir, output_dir, num_samples, include_legend):
        """
        Plot sample annotations with bounding boxes from the cleaned data.
        
        Args:
            src_image_dir (str): Path to the source image directory
            output_dir (str): Path to save the output images
            num_samples (int): Number of samples to plot
            include_legend (bool): Whether to include a legend in the plot
        """
        if self.clean_df is None:
            raise ValueError("No cleaned data available. Run clean_csv_file first.")
            
        df = self.clean_df
        
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
            media_folders = glob.glob(f"{src_image_dir}/*")
            media_folder = [f for f in media_folders if str(media_id) in f][0]
            frame_path = f"{media_folder}/frames/{frame_name}"
            
            if not os.path.exists(frame_path):
                frame_path = f"{media_folder}/{frame_name}/frames/{frame_name}"

            # Make sure the file exists before opening
            if not os.path.exists(frame_path):
                print(f"ERROR: Could not find {frame_name} in {media_folder} folder")
                continue

            # Set the output file name
            output_file = f"{output_dir}/{frame_name}"

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
    Main function to process Zooniverse annotations and optionally plot samples.
    """
    parser = argparse.ArgumentParser(description="Convert Zooniverse Annotations.")

    parser.add_argument("--workflow_id", type=int,
                        default=26428,
                        help="Workflow ID.")

    parser.add_argument("--version", type=float,
                        default=48.28,  # <--
                        help="Version.")

    parser.add_argument("--input_classifications_csv", type=str,
                        help="Path to the input classifications CSV file.")

    parser.add_argument("--output_dir", type=str,
                        default=os.path.abspath("../data/reduced/Season_x"),  # <--
                        help="Path to the label directory.")
    
    parser.add_argument("--plot", action="store_true",
                        help="Include if you want to plot the samples.")
    
    plot_group = parser.add_argument_group('Plotting Options')
    plot_group.add_argument("--src_image_dir", type=str,
                            help="Path to the image directory.")

    plot_group.add_argument("--num_samples", type=int,
                            default=100,
                            help="Number of samples to plot.")

    plot_group.add_argument("--legend_flag", action="store_true",
                            help="Include if user legend should be on the plot")

    args = parser.parse_args()

    # Extract args
    workflow_id = args.workflow_id
    version = args.version
    
    plot_flag = args.plot
    num_samples = args.num_samples
    include_legend = args.legend_flag

    # Extract the shapes for the workflow
    input_classifications_csv = args.input_classifications_csv
    output_dir = f"{args.output_dir}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Validate input CSV exists
    assert os.path.exists(input_classifications_csv), "ERROR: Input CSV provided does not exist"
    
    # Create a ZooniverseProcessor instance and process the data
    processor = ZooniverseProcessor(input_classifications_csv, output_dir, workflow_id, version)
    
    # Set up and validate plotting-related parameters if plotting is requested
    if plot_flag:
        src_image_dir = f"{args.src_image_dir}"
        processor.set_image_directory(src_image_dir)
    
    processor.print_configuration(plot_flag, include_legend)
    
    df, path = processor.clean_csv_file()

    # Plot samples if requested
    if plot_flag:
        processor.plot_samples(processor.src_image_dir, processor.sample_dir, num_samples, include_legend)
        
    print("Done.")


if __name__ == "__main__":
    main()