import os
import shutil
import argparse
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor

import tator
import panoptes_client

import cv2
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------

# TODO
#  This should be updated and made better; although the NAV data isn't perfect, it could still be used to help
#  find frames that are unique and diverse (the user shouldn't see frames that look identical, otherwise
#  they might get bored.
def filter_frames_by_navigation(nav_data, head_thresh=1.0, east_thresh=3.0, north_thresh=3.0):
    """
    :param head_thresh:
    :param east_thresh:
    :param north_thresh:
    :param nav_data:
    :return:
    """
    frames = []

    # Loop through each of the frames' navigational attribute data,
    # calculate the delta between the current frame and the previous
    for fidx, frame in enumerate(nav_data):

        # Skip the first frame (nothing to compare to)
        if fidx == 0:
            frames.append(frame.frame)
            continue

        # Get the heading diff
        curr_head = float(frame.attributes['Heading'])
        prev_head = float(nav_data[fidx - 1].attributes['Heading'])
        diff_head = abs(curr_head - prev_head)

        # Get the easting diff
        curr_east = float(frame.attributes['Eastings'])
        prev_east = float(nav_data[fidx - 1].attributes['Eastings'])
        diff_east = abs(curr_east - prev_east)

        # Get the northing diff
        curr_north = float(frame.attributes['Northings'])
        prev_north = float(nav_data[fidx - 1].attributes['Northings'])
        diff_north = abs(curr_north - prev_north)

        # Find frames that represent where the ROV is moving
        if diff_head >= head_thresh or diff_east >= east_thresh or diff_north >= north_thresh:
            frames.append(frame.frame)

    return frames


# TODO
#   This should be updated and made better; right now it only uses OpenCV to assess the
#   blurriness of an image, however, it's not perfect and many lower quality images
#   still make it through.
def assess_image_quality(image_path, lower_thresh=100, higher_thresh=200):
    """
    :param image_path:
    :param lower_thresh:
    :param higher_thresh:
    :return:
    """
    # Is the image of high quality?
    high_quality = False

    # Load the image using OpenCV, make sure RGB
    image = cv2.imread(image_path)
    # Get the sharpness score
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate the Laplacian variance as a measure of image sharpness
    sharpness = cv2.Laplacian(image, cv2.CV_64F).var()

    # To catch issues with video outage
    if lower_thresh <= sharpness <= higher_thresh:
        high_quality = True

    return high_quality


def download_image(api, media_id, frame, media_dir):
    """

    :param api:
    :param media:
    :param frame:
    :param media_dir:
    :return:
    """
    # Location media directory
    frame_dir = f"{media_dir}/frames"
    os.makedirs(frame_dir, exist_ok=True)

    # Location of file
    path = f"{frame_dir}/{str(frame)}.jpg"

    # If it doesn't already exist, download, move.
    if not os.path.exists(path):
        temp = api.get_frame(media_id, frames=[frame])
        shutil.move(temp, path)

    return path


def upload(client, project, media, dataframe):
    """

    :param client:
    :param project:
    :param media:
    :param dataframe:
    :return:
    """
    try:
        # Create subject set, link to project
        subject_set = client.SubjectSet()
        subject_set.links.project = project
        subject_set.display_name = str(media.id)
        subject_set.save()
        # Reload the project
        project.reload()

        # Convert the dataframe (frame paths) to a dict
        subject_dict = dataframe.to_dict(orient='records')
        # Create a new dictionary with 'Path' as keys and other values as values
        subject_meta = {d['Path']: {k: v for k, v in d.items() if k != 'Path'} for d in subject_dict}

        # Create subjects from the meta
        subjects = []
        subject_ids = []

        # Loop through each of the frames and convert to a subject (creating a subject set)
        for filename, metadata in tqdm(subject_meta.items()):
            # Create the subject
            subject = client.Subject()
            # Link subject to project
            subject.links.project = project
            subject.add_location(filename)
            # Update meta
            subject.metadata.update(metadata)
            # Save
            subject.save()
            # Append
            subjects.append(subject)
            subject_ids.append(subject.id)

        # Add the list of subjects to set
        subject_set.add(subjects)
        # Save the subject set
        subject_set.save()
        project.save()

    except Exception as e:
        raise Exception(f"ERROR: Could not finish uploading subject set for {media.id} to Zooniverse.\n{e}")

    try:
        # Attaching the new subject set to all the active workflows
        workflow_ids = project.__dict__['raw']['links']['active_workflows']

        # If there are active workflows, link them to the next subject sets
        for workflow_id in tqdm(workflow_ids):
            # Create Workflow object
            workflow = client.Workflow(workflow_id)
            workflow_name = workflow.__dict__['raw']['display_name']
            # Add the subject set created previously
            print(f"\nNOTE: Adding subject set {subject_set.display_name} to workflow {workflow_name}")
            workflow.add_subject_sets([subject_set])
            # Save
            workflow.save()
            project.save()

    except Exception as e:
        raise Exception(f"ERROR: Could not link media {media.id} to project workflows.\n{e}")

    # Update the dataframe to now contain the subject IDs
    # This is needed when downloading annotations later.
    dataframe['Subject_ID'] = subject_ids

    return dataframe


def upload_to_zooniverse(args):
    """

    :param args:
    :return:
    """

    print("\n###############################################")
    print("Upload to Zooniverse")
    print("###############################################\n")

    # Root data location
    output_dir = f"{args.output_dir}"
    output_dir = output_dir.replace("\\", "/")
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Login to panoptes using username and password
        panoptes_client.Panoptes.connect(username=args.username, password=args.password)
        print(f"NOTE: Authentication to Zooniverse successful for {args.username}")
    except:
        raise Exception(f"ERROR: Could not login to Panoptes for {args.username}")

    try:
        # Get access to the Zooniverse project given the provided credentials
        project = panoptes_client.Project.find(id=args.zoon_project_id)
        print(f"NOTE: Connected to Zooniverse project '{project.title}' successfully")
    except:
        raise Exception(f"ERROR: Could not access project {args.zoon_project_id}")

    try:
        # Get the TATOR api given the provided token
        api = tator.get_api(host='https://cloud.tator.io', token=args.api_token)
        # Get the correct type of localization for the project (bounding box, attributes)
        tator_project_id = args.tator_project_id
        project_name = api.get_project(id=tator_project_id).name
        state_type_id = 288  # State Type (ROV)
        state_name = api.get_state_type(state_type_id).name
        print(f"NOTE: Authentication to TATOR successful for {api.whoami().username}")
    except Exception as e:
        raise Exception(f"ERROR: Could not obtain needed information from TATOR.\n{e}")

    # --------------------------
    # Download frames from TATOR
    # --------------------------

    # Loop through medias
    for media_id in args.media_ids:

        try:
            # Media name used for output
            media = api.get_media(media_id)
            media_name = media.name
            media_dir = f"{output_dir}/{media_id}"
            os.makedirs(media_dir, exist_ok=True)
            print(f"NOTE: Media ID {media_id} corresponds to {media_name}")

            # Get the frames that have some navigational data instead of downloading all of the frames
            nav_data = api.get_state_list(project=tator_project_id, media_id=[media_id], type=state_type_id)
            print(f"NOTE: Found {len(nav_data)} frames with navigational data for media {media_name}")

            # TODO
            #   Filter out frames based on navigational data?
            #   This could be used to remove multiple sequential frames where there is no movement.
            #   Users shouldn't be shown 1000 frames that look identical: that's boring.
            #   Navigational data might be useful to find more diverse and interesting frames

            frames = filter_frames_by_navigation(nav_data)
            print(f"NOTE: Found {len(frames)} / {len(nav_data)} frames with movement for media {media_name}")

            # Download the frames
            print(f"NOTE: Downloading {len(frames)} frames for {media_name}")
            with ThreadPoolExecutor(max_workers=80) as executor:
                # Submit the jobs
                paths = [executor.submit(download_image, api, media_id, frame, media_dir) for frame in frames]
                # Execute, store paths
                paths = [future.result() for future in tqdm(paths)]

        except Exception as e:
            raise Exception(f"ERROR: Could not finish downloading media {media_id} from TATOR.\n{e}")

        # TODO
        #   Filter frames based on image quality?
        #   Now that the frames have been downloaded, we could asses their quality.
        #   Create a or multiple functions that could be used to determine if a frame
        #   is of lower quality (i.e., blurry) and remove it

        for path in paths:
            # Check the image quality
            if not assess_image_quality(path):
                # Delete the file
                os.remove(path)
                # Remove from the list
                paths.remove(path)

        # TODO
        #   What else can be done to filter out bad frames?
        #   Any machine learning techniques?
        #   Use an existing object detection model?
        #   Use a foundational model?

        # Dataframe for the filtered frames
        dataframe = []

        # Loop through all the filtered frames and store attribute information;
        # this will be used when later when  downloading annotations from
        # Zooniverse after users are done with labeling
        for p_idx, path in enumerate(paths):

            # Add to dataframe
            dataframe.append([media_id,
                              media_name,
                              p_idx,
                              os.path.basename(path),
                              path,
                              media.height,
                              media.width])

        # Output a dataframe to be used for later
        dataframe = pd.DataFrame(dataframe, columns=['Media ID', 'Media Name',
                                                     'Frame ID', 'Frame Name',
                                                     'Path', 'Height', 'Width'])
        if args.upload:
            # ---------------------
            # Upload to Zooniverse
            # ---------------------
            dataframe = upload(panoptes_client, project, media, dataframe)
            dataframe.to_csv(f"{media_dir}/frames.csv", index=False)


# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------

def main():
    """

    """

    parser = argparse.ArgumentParser(description="Upload Data to Zooniverse")

    parser.add_argument("--username", type=str,
                        default=os.getenv('ZOONIVERSE_USERNAME'),
                        help="Zooniverse username")

    parser.add_argument("--password", type=str,
                        default=os.getenv('ZOONIVERSE_PASSWORD'),
                        help="Zooniverse password")

    parser.add_argument("--zoon_project_id", type=int,
                        default=21853,  # click-a-coral
                        help="Zooniverse project ID")

    parser.add_argument("--api_token", type=str,
                        default=os.getenv('TATOR_TOKEN'),
                        help="Tator API Token")

    parser.add_argument("--tator_project_id", type=int,
                        default=70,  # MDBC project
                        help="Tator Project ID")

    parser.add_argument("--media_ids", type=int, nargs='+',
                        help="ID for desired media(s)")

    parser.add_argument("--upload", action='store_true',
                        help="Upload media to Zooniverse (debugging)")

    parser.add_argument("--output_dir", type=str,
                        default=f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/Data',
                        help="Path to the output directory.")

    args = parser.parse_args()

    try:
        upload_to_zooniverse(args)
        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()
