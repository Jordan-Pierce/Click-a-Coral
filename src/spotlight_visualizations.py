import os.path
import argparse
import traceback

import numpy as np
import pandas as pd
from pathlib import Path

from PIL import Image
from renumics import spotlight
import supervision as sv
from ultralytics import YOLO
import cv2
from reduce_annotations import yolo_format


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------


def analyze(data_dir, image_dir, yolo_model, num_samples, chip_image_flag):
    """
    This function uses Renumics Spotlight to visualize the model predictions and user annotations for an image dataset.

    Args:
        data_dir (str): The path to the yolo formatted annotations
        image_dir (str): The path to the image directory
        yolo_model (str): The path to the pre-trained model to be used as weights
        num_samples (int): The number of media folders to loop through
        chip_image_flag (bool): Whether to use the whole image or the chipped images in visualizations
    """

    # Initialize dataframe
    df = pd.DataFrame(columns=['filepath', 'categories', 'bboxs'])

    if chip_image_flag:
        chip_df = pd.DataFrame(columns=['chip_path', 'class', 'bbox'])
        chip_dir = "\Kira\GitHub\Zooniverse\Chip_Images"
        yolo_chip_dir = "\Kira\GitHub\Zooniverse\Yolo_Chip_Images"
        yolo_chip_df = pd.DataFrame(columns=['yolo_chip_path', 'yolo_class', 'yolo_bbox'])
        os.makedirs(chip_dir, exist_ok=True)
        os.makedirs(yolo_chip_dir, exist_ok=True)

    count = 0

    # Initializes dicts for YOLO convert later
    detections = {}
    images = {}

    for dir in os.listdir(data_dir):

        dir = f"{data_dir}\\{dir}"

        classes = pd.read_csv(f"{dir}\\classes.txt", names=['Class'])
        classes = classes['Class'].unique().tolist()

        # Get media folder
        media_folder = os.path.basename(dir)

        dir = f"{dir}\\train\\labels"

        for file in os.listdir(dir):

            filepath = f"{dir}\\{file}"

            image = os.path.splitext(file)[0]
            image_path = f"{image_dir}\\{media_folder}\\frames\\{image}.jpg"

            # Open and read txt file
            bbox_df = pd.read_csv(filepath, sep=" ", names=['Class', 'x1', 'y1', 'x2', 'y2'])

            categories = []
            bboxs = []

            for i, row in bbox_df.iterrows():
                class_num = int(row['Class'])
                class_name = classes[class_num]
                categories.append(class_name)

                x_center, y_center, w, h = row['x1'], row['y1'], row['x2'], row['y2']

                x1 = x_center - (w/2)
                y1 = y_center - (h/2)
                x2 = x1 + w
                y2 = y1 + h

                bbox = [x1, y1, x2, y2]

                # OPTIONAL
                if chip_image_flag:
                    chip_path, chip_name = chip_image(image_path, class_name, x1, y1, x2, y2, chip_dir, media_folder, image, i)

                    if chip_path is None:
                        break
                    chip_row = {'chip_path': chip_path, 'class': class_name, 'bbox': [0, 0, 0, 0]}
                    chip_df = chip_df._append(chip_row, ignore_index=True)

                    # Adds image to dict for YOLO convert
                    im = cv2.imread(chip_path)
                    height, width, channel = im.shape
                    images[chip_name] = im

                    xyxy = np.empty((1, 4))
                    xyxy[0] = [0, 0, width, height]

                    detection = sv.Detections(xyxy=xyxy, class_id=np.array([class_num]))

                    # Adds detection to dict
                    detections[chip_name] = detection

                bboxs.append(bbox)

            new_row = {'filepath': image_path, 'categories': categories, 'bboxs': bboxs}
            df = df._append(new_row, ignore_index=True)

        count += 1
        if num_samples == count:
            break

    if chip_image_flag:
        # Creates YOLO directories
        yolo_dir = f"{chip_dir}\\Yolo"
        train_dir = f"{yolo_dir}\\train"
        test_dir = f"{yolo_dir}\\test"

        os.makedirs(yolo_dir, exist_ok=True)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # Creates Detection Dataset
        cd = sv.ClassificationDataset(classes=classes, images=images, annotations=detections)

        # Split to training and validation sets
        train_cd, test_cd = cd.split()

        # Converts to YOLO format
        train_cd.as_folder_structure(train_dir)
        test_cd.as_folder_structure(test_dir)



    # Do the yolo stuff
    detection_model = YOLO(yolo_model)

    chip_detections = {}
    chip_images = {}

    detections = []
    for filepath in df["filepath"].tolist():
        detection = detection_model(filepath)[0]

        detections.append(
            {
                "yolo_bboxs": [np.array(box.xyxyn.tolist())[0] for box in detection.boxes],
                "yolo_conf": np.mean([np.array(box.conf.tolist())[0] for box in detection.boxes]),
                "yolo_categories": np.array(
                    [np.array(detection.names[int(box.cls)]) for box in detection.boxes]
                ),
            }
        )

        if chip_image_flag:
            # Get media folder
            path_parts = Path(filepath).parts
            media_folder = path_parts[5]
            image_name = f"{path_parts[-1].split('.')[0]}"

            i = 0
            for box in detection.boxes:
                bbox = box.xyxyn.tolist()
                x1, y1, x2, y2 = bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3]
                class_name = detection.names[int(box.cls)]
                yolo_chip_path, yolo_chip_name = chip_image(filepath, class_name, x1, y1, x2, y2, yolo_chip_dir, media_folder, image_name, i)

                yolo_chip_row = {'yolo_chip_path': yolo_chip_path, 'yolo_class': class_name, 'yolo_bbox': [0, 0, 0, 0]}

                yolo_chip_df = yolo_chip_df._append(yolo_chip_row, ignore_index=True)

                # Adds image to dict for YOLO convert
                im = cv2.imread(yolo_chip_path)
                height, width, channel = im.shape
                chip_images[yolo_chip_name] = im

                xyxy = np.empty((1, 4))
                xyxy[0] = [0, 0, width, height]

                chip_detection = sv.Detections(xyxy=xyxy, class_id=np.array([int(box.cls)]))

                # Adds detection to dict
                chip_detections[yolo_chip_name] = chip_detection

                i += 1

    # Creates YOLO directories
    yolo_dir = f"{yolo_chip_dir}\\Yolo"
    train_dir = f"{yolo_dir}\\train"
    test_dir = f"{yolo_dir}\\test"

    os.makedirs(yolo_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Creates Detection Dataset
    cd = sv.ClassificationDataset(classes=classes, images=chip_images, annotations=chip_detections)

    # Split to training and validation sets
    train_cd, test_cd = cd.split()

    # Converts to YOLO format
    train_cd.as_folder_structure(train_dir)
    test_cd.as_folder_structure(test_dir)

    df_yolo = pd.DataFrame(detections)

    if chip_image_flag:
        chip_merged = pd.concat([chip_df, yolo_chip_df], axis=1)
        spotlight.show(chip_merged, embed=['yolo_chip_path', 'chip_path'])
    else:
        df_merged = pd.concat([df, df_yolo], axis=1)
        spotlight.show(df_merged, embed=['filepath'])


def chip_image(image_path, class_name, x1, y1, x2, y2, chip_dir, media_folder, image_name, i):
    """
    This function chips an image based on the bounding boxe coordinates.

    Args:
        image_path (str): The filepath to the whole image
        class_name (str): The classification for the bounding box
        x1 (float): The normalized top-left x-coordinate for the bounding box
        y1 (float): The normalized top-left y-coordinate for the bounding box
        x2 (float): The normalized bottom-right x-coordinate for the bounding box
        y2 (float): The normalized bottom-right y-coordinate for the bounding box
        chip_dir (str): The root path to the directory for the chipped images
        media_folder (str): The media folder of the image
        image_name (str): The name of the .jpg image
        i (int): The index for the bounding box of an image

    Returns: New filepath to the chipped image
    """

    # Make a new directory for the image chips
    path = f"{chip_dir}\\{media_folder}"
    os.makedirs(path, exist_ok=True)

    # Open the original image
    im = Image.open(image_path)

    # Find the width and height of the image
    width, height = im.size

    # Crop the image and un-normalize the coordinates
    #TODO: Need to fix this (expecting int getting float, or find a way to pass the coordinates in un-normalized)
    im1 = im.crop(((x1 * width), (y1 * height), (x2 * width), (y2 * height)))

    # Check that the chip does not return an empty image
    if im1.width and im1.height != 0:

        chip_name = f"{image_name}_{class_name}-{i}.jpg"

        # Save the image
        im1.save(f"{path}\\{chip_name}")

        # Return the filepath to the chipped image
        return f"{path}\\{chip_name}", chip_name
    else:
        return None, None


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """

    """

    parser = argparse.ArgumentParser(description="Visualize data in Spotlight")

    parser.add_argument("--data", type=str,
                        default="./Reduced/Yolo",
                        help="Data directory")

    parser.add_argument("--image_dir", type=str,
                        default="./Data")

    parser.add_argument("--yolo_model", type=str,
                        default="./Reduced/Runs/2024-07-02_09-11-56_detect_yolov10m/weights/best.pt")

    parser.add_argument("--num_samples", type=int,
                        default=20,
                        help="The number of media ID's to include")

    parser.add_argument("--chip_image", action="store_true",
                        help="If the image should be chipped or not")

    args = parser.parse_args()

    dataset = args.data
    image_dir = args.image_dir
    yolo_model = args.yolo_model
    num_samples = args.num_samples
    chip_image_flag = args.chip_image

    try:

        analyze(dataset, image_dir, yolo_model, num_samples, chip_image_flag)

        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()