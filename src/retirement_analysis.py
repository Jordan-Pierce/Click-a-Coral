import os
import argparse
import traceback

import pandas as pd
import matplotlib.pyplot as plt

from visualize import create_image_row
from reduce_annotations import save_to_csv, get_now
from reduce_annotations import get_image, remove_single_clusters

# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def annotation_count(retirement_dir, output_dir, image_dir, iou_threshold):
    """
    This function compares the number of original and reduced annotations at different retirement ages

    Args:
        retirement_dir (str): Path to a retirement directory containing reduced annotations and original annotations
        at different retirement ages
        output_dir (str): Path to where the results should be outputted
        image_dir (str): Path to the image directory
        iou_threshold (float): The IoU threshold that should be considered for mAP
    """

    # Initialize dataframe
    counts = pd.DataFrame(columns=['Retirement_Age', 'Reduced_Annotation_Number', 'Extracted_Annotation_Number'])

    # Run through the retirement directory
    for directory in os.listdir(retirement_dir):

        # Makes sure the nested folder is a retirement folder
        if directory.startswith("Retirement"):

            # Gets the retirement age
            age = int(directory.split("_", 1)[1])

            directory = f"{retirement_dir}\\{directory}\\Reduced"

            # Finds the reduced and original annotation csv files
            reduced = [file for file in os.listdir(directory) if file.startswith("reduced")][0]
            extracted = [file for file in os.listdir(directory) if file.startswith("extracted")][0]

            # Turns the csv into a dataframe
            reduced = pd.read_csv(f"{directory}\\{reduced}")
            extracted = pd.read_csv(f"{directory}\\{extracted}")

            # Gets the image dataframe
            image_df = get_image_df(reduced, extracted, image_dir, iou_threshold, age)

            # Create output CSV file for reduced annotations
            output_csv = f"{output_dir}\\Retirement_{age}\\image_info"

            # Get the timestamp
            timestamp = get_now()

            # Save the csv
            save_to_csv(output_csv, image_df, False, timestamp)

            # Get the number of reduced and original annotations
            reduced_number = len(reduced)
            extracted_number = len(extracted)

            # Add to dataframe
            new_row = {'Retirement_Age': age, 'Reduced_Annotation_Number': reduced_number,
                       'Extracted_Annotation_Number': extracted_number}
            counts = counts._append(new_row, ignore_index=True)

        else:
            continue

    # Sort by retirement age
    counts.sort_values(by='Retirement_Age')

    # Initialize the plot
    fig, axs = plt.subplots(ncols=2, figsize=(20, 10))

    # Plot the left subplot for reduced annotations
    axs[0].bar(counts['Retirement_Age'], counts['Reduced_Annotation_Number'], width=3)
    axs[0].set_title("Reduced Annotations")

    # Plot right subplot for original annotations
    axs[1].bar(counts['Retirement_Age'], counts['Extracted_Annotation_Number'], width=3)
    axs[1].set_title("Original Annotations")

    # Set axis labels
    fig.supylabel("Number of Annotations")
    fig.supxlabel("Retirement Age")

    plt.tight_layout()

    # Save the plot
    plt.savefig(f"{output_dir}\\annotation_count_comparison.jpg", bbox_inches='tight')
    plt.close()


def get_image_df(reduced_annotations, extracted_annotations, image_dir, iou_threshold, retirement_age):
    """
    This function gets an image dataframe containing information about the image mAP, IoU, etc

    Args:
        reduced_annotations (Pandas dataframe): A reduced annotation dataframe
        extracted_annotations (Pandas dataframe): The extracted original annotation dataframe
        image_dir (str): The path to the image directory
        iou_threshold (float): The IoU threshold to be used for calculating mAP
        retirement_age (int): The retirement age of the dataframe
        
    Returns:
        metrics_df (Pandas dataframe): A dataframe giving metric information for the images present in the annotated 
        dataframe
    """

    # Make the subject ID's integers
    extracted_annotations['Subject ID'] = extracted_annotations['Subject ID'].astype(int)
    reduced_annotations['Subject ID'] = reduced_annotations['Subject ID'].astype(int)

    # Group both dataframes by subject ID
    extracted_groups = extracted_annotations.groupby("Subject ID")

    # Initialize metrics dataframe
    metrics_df = pd.DataFrame(columns=['Subject ID', 'Average_IoU', 'Precision',
                                       'Recall', 'image_path', 'Number_of_Annotations', 'Mean_Average_Precision'])

    # Loop through the subject IDs in the original annotations
    for subject_id, subject_id_df in extracted_groups:

        print("Subject ID", subject_id)
        print("Subject Annotations", subject_id_df)

        # Get image information
        image_path, image_name, frame_name = get_image(subject_id_df.iloc[0], image_dir)

        # Get the reduced annotations for the subject ID
        if (reduced_annotations['Subject ID'] == subject_id).any():

            # Get new dataframe
            no_singles = remove_single_clusters(subject_id_df)

            metrics_df = create_image_row(no_singles, subject_id, metrics_df, image_path, 
                                          image_name, iou_threshold, retirement_age)
        else:
            continue

    return metrics_df


def metric_comparison(retirement_dir, output_dir):
    """
    This function plots and compares the average number of annotations, IoU, and mAP for different retirement ages
    
    Args:
        retirement_dir (str): Path to the retirement directory
        output_dir (str): Path to the output directory where the plots should be saved
    """

    # Initialize dataframe
    retirement_metrics = pd.DataFrame(columns=['Retirement_Age', 'Number_of_Images',
                                               'Average_Annotations', 'Average_IoU', 'mAP'])

    # Loop through the retirement folders
    for directory in os.listdir(retirement_dir):

        # Check that folder starts with retirement
        if directory.startswith("Retirement"):

            # Get the retirement age
            age = int(directory.split("_", 1)[1])

            directory = f"{retirement_dir}\\{directory}"

            # Get the associated image info csv file
            image_info = [file for file in os.listdir(directory) if file.startswith("image")][0]

            # Change the csv into a dataframe 
            csv_path = f"{directory}\\{image_info}"
            image_info = pd.read_csv(csv_path)

            # Get the number of images
            image_number = len(image_info)

            # Get the average number of annotations per image
            average_annotation_num = image_info['Number_of_Annotations'].mean()

            # Get the average IoU for all images
            average_iou = image_info['Average_IoU'].mean()

            # Get the mAP of all the images
            map = image_info['Mean_Average_Precision'].mean()

            # Add information to retirement metrics dataframe
            new_row = {'Retirement_Age': age, 'Number_of_Images': image_number,
                       'Average Number of Annotations': average_annotation_num, 'Average IoU': average_iou, 'mAP': map}
            retirement_metrics = retirement_metrics._append(new_row, ignore_index=True)
        else:
            continue

    # Sort by retirement age
    retirement_metrics.sort_values(by='Retirement_Age')

    # Make a list of what the y-axis should be
    y_values = ['Average Number of Annotations', 'Average IoU', 'mAP']

    # Make the plots
    for y in y_values:

        # Initialize the plot
        plt.figure(figsize=(20, 10))

        # Plot the left subplot for reduced annotations
        plt.bar(retirement_metrics['Retirement_Age'], retirement_metrics[y], width=3)

        # Set axis labels
        plt.xlabel("Retirement Age")
        plt.ylabel(f"{y} per Image")

        plt.tight_layout()

        # Save the plot
        plt.savefig(f"{output_dir}\\{y}.jpg", bbox_inches='tight')
        plt.close()


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """

    """

    parser = argparse.ArgumentParser(description="Determine the minimum retirement age for images in Zooniverse")

    parser.add_argument("--retirement_dir", type=str,
                        default="./Retirement_Analysis",
                        help="The directory comparing retirement ages to compare")

    parser.add_argument("--output_dir", type=str,
                        default="./Retirement_Analysis",
                        help="Output directory")

    parser.add_argument("--image_dir", type=str,
                        default="./Data",
                        help="Image directory")

    parser.add_argument("--iou_threshold", type=float,
                        default=0.6,
                        help="The IoU threshold to be used for mAP")

    args = parser.parse_args()

    # Parse out arguments
    retirement_dir = args.retirement_dir
    output_dir = args.output_dir
    image_dir = args.image_dir
    iou_threshold = args.iou_threshold

    try:

        annotation_count(retirement_dir, output_dir, image_dir, iou_threshold)

        metric_comparison(retirement_dir, output_dir)

        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()