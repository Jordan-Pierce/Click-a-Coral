import os
import glob
import shutil
import argparse
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor

import tator
import panoptes_client

import utm
import math
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


# ------------------------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------------------------

def filter_frames_by_navigation(nav_data, dist_thresh=1.5):
    """
    :param dist_thresh: Threshold distance in meters to consider a new point
    :param nav_data:
    :return:
    """
    frames = []
    prev_point = None

    # Loop through each of the frames' navigational attribute data
    for fidx, frame_data in enumerate(nav_data):

        # Get the current easting and northing using .get(), default to None
        curr_east = frame_data.attributes.get('Eastings_Raw', None)
        curr_north = frame_data.attributes.get('Northings_Raw', None)

        # If either is None, skip this frame
        if curr_east is None or curr_north is None:
            # Check for latitude and longitude
            curr_lat = frame_data.attributes.get('Latitude', None)
            curr_lon = frame_data.attributes.get('Longitude', None)
            if curr_lat is not None and curr_lon is not None:
                try:
                    utm_result = utm.from_latlon(float(curr_lat), float(curr_lon))
                    curr_east, curr_north = utm_result[0], utm_result[1]
                except Exception:
                    continue
            else:
                continue

        curr_east = float(curr_east)
        curr_north = float(curr_north)
        curr_point = (curr_east, curr_north)

        # For the first frame, add it to the list
        if fidx == 0:
            frames.append(frame_data.frame)
            prev_point = curr_point
            continue

        # Calculate the distance between the current point and the previous point
        distance = math.sqrt((curr_east - prev_point[0])**2 + (curr_north - prev_point[1])**2)

        # If the distance is greater than the threshold, add the current frame
        if distance >= dist_thresh:
            frames.append(frame_data.frame)
            prev_point = curr_point

    return frames


def download_image(api, media_id, frame, frame_dir):
    """

    :param api:
    :param media:
    :param frame:
    :param frame_dir:
    :return:
    """
    # Location of file
    path = f"{frame_dir}/{str(frame)}.jpg"

    # If it doesn't already exist, download, move.
    if not os.path.exists(path):
        temp = api.get_frame(media_id, frames=[frame])
        shutil.move(temp, path)

    return path


def upload(client, project, media, dataframe, set_active):
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

    if set_active:

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
    except Exception as e:
        raise Exception(f"ERROR: Could not login to Panoptes for {args.username}\n{e}")

    try:
        # Get access to the Zooniverse project given the provided credentials
        project = panoptes_client.Project.find(id=args.zoon_project_id)
        print(f"NOTE: Connected to Zooniverse project '{project.title}' successfully")
    except Exception as e:
        raise Exception(f"ERROR: Could not access project {args.zoon_project_id}.\n{e}")

    try:
        # Get the TATOR api given the provided token
        api = tator.get_api(host='https://cloud.tator.io', token=args.api_token)
        # Get the correct type of localization for the project (bounding box, attributes)
        tator_project_id = args.tator_project_id
        state_type_id = 288  # State Type (ROV)
        print(f"NOTE: Authentication to TATOR successful for {api.whoami().username}")
    except Exception as e:
        raise Exception(f"ERROR: Could not obtain needed information from TATOR.\n{e}")
    
    if not args.existing_csv:

        # ---------------------------
        # Download frames from TATOR
        # ---------------------------

        # Loop through medias
        for media_id in args.media_ids:

            try:
                # Media name used for output
                media = api.get_media(media_id)
                media_name = media.name
                media_dir = f"{output_dir}/{media_id}"
                frame_dir = f"{media_dir}/frames"

                if os.path.exists(frame_dir):
                    raise Exception(f"ERROR: Frames for media {media_id} already exist in {frame_dir}; "
                                    f"this media has already been uploaded.")

                os.makedirs(frame_dir)
                print(f"NOTE: Media ID {media_id} corresponds to {media_name}")

                # Get the frames that have some navigational data instead of downloading all of the frames
                nav_data = api.get_state_list(project=tator_project_id, media_id=[media_id], type=state_type_id)
                if len(nav_data) == 0:
                    raise Exception(f"ERROR: No navigational data found for media {media_name} (ID: {media_id}). "
                                    f"This media might not have navigational data.")
                    
                print(f"NOTE: Found {len(nav_data)} frames with navigational data for media {media_name}")

                frames = filter_frames_by_navigation(nav_data, dist_thresh=args.dist_thresh)
                if len(frames) == 0:
                    raise Exception(f"ERROR: No frames found with movement for media {media_name} (ID: {media_id}). "
                                    f"Please check the distance threshold or the navigational data.")
                    
                print(f"NOTE: Found {len(frames)} / {len(nav_data)} frames with movement for media {media_name}")

                # Download the frames
                print(f"NOTE: Downloading {len(frames)} frames for {media_name}")
                with ThreadPoolExecutor() as executor:
                    # Submit the jobs
                    paths = [executor.submit(download_image, api, media_id, frame, frame_dir) for frame in frames]
                    # Execute, store paths
                    paths = [future.result() for future in tqdm(paths)]

            except Exception as e:
                raise Exception(f"ERROR: Could not finish downloading media {media_id} from TATOR.\n{e}")

            # -------------------------------------
            # Manual Removal of Low Quality Frames
            # -------------------------------------
            _ = input("NOTE: Remove low quality frames manually; press 'Enter' when finished.")
            # Get the remaining high quality frames
            paths = glob.glob(f"{frame_dir}/*.jpg")

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
    else:
        try:
            # Load the existing CSV file
            dataframe = pd.read_csv(args.existing_csv)
            media_id = str(dataframe['Media ID'].unique()[0])
            media = api.get_media(media_id)
            media_name = media.name
            media_dir = f"{output_dir}/{media_id}"
            dataframe['Path'] = [f"{media_dir}/frames/{os.path.basename(p)}" for p in dataframe['Frame Name']]
            print(f"NOTE: Loaded existing CSV file with {len(dataframe)} media")
        except Exception as e:
            raise Exception(f"ERROR: Could not load existing CSV file {args.existing_csv}\n{e}")

    if args.upload:
        # ---------------------
        # Upload to Zooniverse
        # ---------------------
        set_active = args.set_active

        dataframe = upload(panoptes_client, project, media, dataframe, set_active)
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
    
    parser.add_argument("--existing_csv", type=str, default=None,
                        help="Path to existing CSV file with media information")

    parser.add_argument("--media_ids", type=int, nargs='+',
                        help="ID for desired media(s)")

    parser.add_argument("--dist_thresh", type=float,
                        default=1.5,
                        help="The distance (m) between successive frames to sample")

    parser.add_argument("--set_active", action='store_true',
                        help="Make subject-set active with current workflow")

    parser.add_argument("--upload", action='store_true',
                        help="Upload media to Zooniverse (debugging)")

    parser.add_argument("--output_dir", type=str,
                        default=f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/data/curated',
                        help="Path to the output directory.")

    args = parser.parse_args()

    try:
        upload_to_zooniverse(args)
        print("Done.")

    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()