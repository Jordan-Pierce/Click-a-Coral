import os.path
import argparse
import traceback
from tqdm import tqdm

import numpy as np
import pandas as pd
from pathlib import Path

import cv2
import supervision as sv
from ultralytics import YOLO
from renumics import spotlight


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def xywh_to_xyxyn(bbox):
    """convert from xywh to xyxyn format"""
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


def run_spotlight(yolo_dir, yolo_model=None):
    """
    This function uses Renumics Spotlight to visualize the model predictions and user annotations for an image dataset.

    Args:
        yolo_dir (str): The path to the yolo-formatted object detect dataset
        yolo_model (str): The path to the pre-trained model to be used as weights
    """
    # Initialize dataframe
    df = pd.DataFrame(columns=['filepath', 'categories', 'bboxs'])

    dataset = []
    
    # User Supervision to load the YOLO dataset
    for subfolder in ["train", "valid", "test"]:
        # Load the YOLO dataset
        data_path = f"{yolo_dir}/{subfolder}"
        
        if not os.path.exists(data_path):
            continue
        
        print(f"Loading {subfolder} dataset from {data_path}")
        data = sv.DetectionDataset.from_yolo(images_directory_path=f"{data_path}/images",
                                             annotations_directory_path=f"{data_path}/labels",
                                             data_yaml_path=f"{yolo_dir}/data.yaml")
        
        dataset.append(data)
        
    # Merge the datasets
    dataset = sv.DetectionDataset.merge(dataset)
    
    row = []
    for path, image, detections in tqdm(dataset, desc="Loading images and annotations"):
        # Get the class names for each of the annotations
        labels = [dataset.classes[class_id] for class_id in detections.class_id]
        # Get the bounding boxes for each of the annotations
        bboxs = [xywh_to_xyxyn(detection.astype(int)) for detection in detections.xyxy]
        row.append([path, labels, bboxs])

    # Create a DataFrame from the loaded data
    df = pd.DataFrame(row, columns=["filepath", "categories", "bboxs"])
        
    if yolo_model is not None:
        # Load the YOLO model
        detection_model = YOLO(yolo_model)

        # Goes through all the detections
        for filepath in df["filepath"].tolist():
            detection = detection_model(filepath)[0]

            # Adds detection information
            detections.append(
                {
                    "yolo_bboxs": [np.array(box.xyxyn.tolist())[0] for box in detection.boxes],
                    "yolo_conf": np.mean([np.array(box.conf.tolist())[0] for box in detection.boxes]),
                    "yolo_categories": np.array(
                        [np.array(detection.names[int(box.cls)]) for box in detection.boxes]
                    ),
                }
            )

        df = pd.concat([df, pd.DataFrame(detections)], axis=1)

    # Shows in spotlight
    print("Starting Spotlight...")
    spotlight.show(df, embed=['filepath'])


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------


def main():
    """

    """

    parser = argparse.ArgumentParser(description="Visualize data in Spotlight")

    parser.add_argument("--yolo_dir", type=str, required=True,
                        help="YOLO dataset directory")

    parser.add_argument("--yolo_model", type=str,
                        help="Path to the YOLO model weights")

    args = parser.parse_args()

    yolo_dir = args.yolo_dir
    yolo_model = args.yolo_model

    try:
        assert os.path.exists(yolo_dir), f"Directory {yolo_dir} does not exist."
        
        # Run the analysis
        run_spotlight(yolo_dir, yolo_model)
        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()