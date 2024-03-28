import os
import shutil
import traceback
import concurrent.futures

import numpy as np

import tator
import supervision as sv
from ultralytics import YOLO


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------

def filter_detections(image, annotations, area_thresh=0.005, conf_thresh=0.0):
    """

    :param image:
    :param annotations:
    :param area_thresh:
    :param conf_thresh:
    :return annotations:
    """

    height, width, channels = image.shape
    image_area = height * width

    # Filter by relative area first
    annotations = annotations[(annotations.box_area / image_area) >= area_thresh]

    # Get the box areas
    boxes = annotations.xyxy
    num_boxes = len(boxes)

    is_large_box = np.zeros(num_boxes, dtype=bool)

    for i, box1 in enumerate(boxes):
        x1, y1, x2, y2 = box1

        # Count the number of smaller boxes contained within box1
        contained_count = np.sum(
            (boxes[:, 0] >= x1) &
            (boxes[:, 1] >= y1) &
            (boxes[:, 2] <= x2) &
            (boxes[:, 3] <= y2)
        )

        # Check if box1 is a large box containing at least two smaller boxes
        is_large_box[i] = contained_count >= 3

    annotations = annotations[~is_large_box]

    # Filter by confidence
    annotations = annotations[annotations.confidence > conf_thresh]

    return annotations


def infer(media_dir, conf=0.15, iou=0.3, debug=True):
    """

    """
    # Raw frame location (from TATOR)
    downloaded_frames_dir = f"{media_dir}/frames"
    os.makedirs(downloaded_frames_dir, exist_ok=True)

    # Rendered frames with detections
    rendered_frames_dir = f"{media_dir}/rendered_frames"
    os.makedirs(rendered_frames_dir, exist_ok=True)

    # Get the root directory
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    root = root.replace("\\", "/")

    # Model location
    model_path = f"{root}/Models/best.pt"

    if not os.path.exists(model_path):
        raise Exception(f"ERROR: Model weights not found in {root}!")

    try:
        # Load it up
        model = YOLO(model_path)
        print(f"NOTE: Successfully loaded weights {model_path}")

        # Create the annotators
        box_annotator = sv.BoundingBoxAnnotator()
        percentage_bar_annotator = sv.PercentageBarAnnotator()

    except Exception as e:
        raise Exception(f"ERROR: Could not load model weights {model_path}.\n{e}")

    # -------------------------------------------
    # Make Inferences
    # -------------------------------------------

    try:
        print("NOTE: Performing inference")

        # Generator for model performing inference
        results = model(f"{downloaded_frames_dir}",
                        conf=conf,
                        iou=iou,
                        augment=False,
                        max_det=2000,
                        verbose=False,
                        stream=True)  # generator of Results objects

        # Loop through the results
        with sv.ImageSink(target_dir_path=rendered_frames_dir, overwrite=True) as sink:
            for result in results:

                # Version issue
                result.obb = None

                # Original frame
                frame_name = os.path.basename(result.path)
                original_frame = result.orig_img

                # Convert the results
                detections = sv.Detections.from_ultralytics(result)

                if debug:
                    # Create rendered results
                    annotated_frame = box_annotator.annotate(scene=original_frame, detections=detections)
                    annotated_frame = percentage_bar_annotator.annotate(scene=annotated_frame, detections=detections)

                    # Save to render folder
                    sink.save_image(image=annotated_frame, image_name=frame_name)
                    print(f"NOTE: Rendered results for {frame_name}")

    except Exception as e:
        print(f"ERROR: {e}")
        print(traceback.print_exc())

    return results