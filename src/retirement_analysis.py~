import os
import random
import argparse

import traceback
import numpy as np
import pandas as pd
from datetime import datetime as dt

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn.metrics import auc
from reduce_annotations import get_image, remove_single_clusters

# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def annotation_count(retirement_dir):

    # Initialize dataframe
    counts = pd.DataFrame(columns=['Retirement_Age', 'Reduced_Annotation_Number', 'Extracted_Annotation_Number'])

    for dir in os.listdir(retirement_dir):

        age = dir.split("_", 1)[1]

        dir = f"{retirement_dir}\\{dir}\\Reduced"

        reduced = [file for file in os.listdir(dir) if file.startswith("reduced")][0]

        extracted = [file for file in os.listdir(dir) if file.startswith("extracted")][0]

        reduced = pd.read_csv(f"{dir}\\{reduced}")
        extracted = pd.read_csv(f"{dir}\\{extracted}")

        reduced_number = len(reduced)
        extracted_number = len(extracted)

        # Add to dataframe
        new_row = {'Retirement_Age': age, 'Reduced_Annotation_Number': reduced_number,
                   'Extracted_Annotation_Number': extracted_number}

        counts = counts._append(new_row, ignore_index=True)

    print(counts)



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
                        default="./Test",
                        help="Output directory")

    args = parser.parse_args()

    # Parse out arguments
    retirement_dir = args.retirement_dir
    output_dir = args.output_dir

    try:

        annotation_count(retirement_dir)

        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()