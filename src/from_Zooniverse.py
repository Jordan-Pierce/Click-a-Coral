import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from panoptes_aggregation.csv_utils import unjson_dataframe


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------
# TODO
#   Create functions that actually aggregate / reduce the results from multiple users for each subject (frame)
#   Aggregation / reduction is somewhat heuristic, so look into techniques that help with this (including ML methods).
#   The end goal is that we export aggregated / reduced results into YOLO format, which we can use for model training.
#   See here for more details on how to do this using panoptes-aggregate
#   https://aggregation-caesar.zooniverse.org/How_Clustering_Works.html

def run_shape_extractor():
    """

    :return:
    """


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------
# TODO
#   This should be converted into a script that be ran via CMD using argparse
#   See to_Zooniverse.py for how it should be set up

# Input and output folders
image_folder = f"./Data/images"
output_folder = f"./Data/workflow"
os.makedirs(output_folder, exist_ok=True)

# TODO
#   Run a sub-process to extract the shapes (rectangles) as a csv file.
#   There is a way to do this via GUI or CMD (but not python?) So use
#   sub-process to do this within the script.
#   See here for more details on how to do this using panoptes-aggregate
#   https://aggregation-caesar.zooniverse.org/Scripts.html

# Extract the shapes for the workflow
run_shape_extractor()

# Path to the extracted csv file
csv_path = "Data/aggregation/shape_extractor_rectangle_workflow.csv"
df = pd.read_csv(csv_path)
unjson_dataframe(df)

# The current media
media_id = None
media_csv = f"./Data/{media_id}/frame.csv"
media_df = pd.read_csv(media_csv)

# To record results for all users activity
all_users_results = {}

# Loop through all the frames in dataset
for frame_id in tqdm(media_df['Frame ID'].unique()):

    # Get the mapping from frame to subject id
    subject_id = media_df[media_df['Frame ID'] == frame_id]['Subject ID'].values[0]
    df_sub = df[df['Subject ID'] == subject_id]
    df_sub = df_sub.dropna()

    # Get all users boxes for current frame
    x_lists = df_sub['data.frame0.T0_tool0_x'].values
    y_lists = df_sub['data.frame0.T0_tool0_y'].values
    w_lists = df_sub['data.frame0.T0_tool0_width'].values
    h_lists = df_sub['data.frame0.T0_tool0_height'].values

    # Get usernames
    users_frame = df_sub['user_name'].values
    num_users_frame = len(users_frame)

    if not num_users_frame:
        continue

    # Plot the data
    fig, ax = plt.subplots(figsize=(20, 10))

    # Lists to store information for the legend
    legend_labels = []
    edge_patches = []
    boxes_per_frame = []

    # Plot bounding boxes
    for user_idx, (x_list, y_list, w_list, h_list) in enumerate(zip(x_lists, y_lists, w_lists, h_lists)):

        # Generate a random RGB color for each user
        edge_color = tuple(np.random.rand(3, ))
        # Tally for user boxes
        user_num_boxes = 0

        # Loop through each users' boxes
        for x, y, w, h in zip(x_list, y_list, w_list, h_list):

            # Must be a valid box
            if w and h:

                # Plot each box the user made
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=edge_color, facecolor='none')
                ax.add_patch(rect)

                # Tally the number of valid boxes
                user_num_boxes += 1

                # Tally for the number of boxes made per user for all frames in dataset
                if users_frame[user_idx] in all_users_results:
                    all_users_results[users_frame[user_idx]] += 1
                else:
                    all_users_results[users_frame[user_idx]] = 1

        # Add to the list of boxes made per user
        boxes_per_frame.append(user_num_boxes)

        # Store information for the legend
        legend_labels.append(f'User {users_frame[user_idx]} - {user_num_boxes} boxes')
        edge_patch = patches.Patch(color=edge_color, label=f'User {user_idx + 1}')
        edge_patches.append(edge_patch)

    # Add legend outside the plot
    ax.legend(handles=edge_patches, labels=legend_labels, loc='upper left', bbox_to_anchor=(1, 1))

    # Total boxes and average boxes per frame
    total_boxes = np.sum(boxes_per_frame).astype(int)
    average_boxes = np.mean(boxes_per_frame).astype(int)

    # Show the plot
    plt.title(f"Frame: {frame_id}    "
              f"Total boxes: {total_boxes}    "
              f"Average boxes: {average_boxes}")

    # Save with same name as frame in results folder
    plt.imshow(plt.imread(f"{image_folder}/{frame_id}.png"))
    plt.savefig(f"{output_folder}/{frame_id}.png", bbox_inches='tight')
    # plt.show()
    plt.close()

# This plots all the user's data as a bar chart
df = pd.DataFrame(list(all_users_results.items()), columns=['Category', 'Value'])

if not df.empty:
    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    # Create a gradient color map for each bar based on values
    colors = plt.cm.viridis(df['Value'] / df['Value'].max())
    # Plotting the bar chart with gradient colors
    bars = plt.bar(df.index, df['Value'], color=colors)
    # Adding labels and title
    plt.xlabel('User')
    plt.ylabel('Count')
    plt.title('User Annotations')
    # Remove x-axis ticks and labels
    plt.xticks([])
    # Display the plot
    plt.savefig(f"{output_folder}/AllUsers.png")
    plt.show()
    plt.close()

print("Done.")
