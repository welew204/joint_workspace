import cv2
import poseDetectionModule as pm
import time
import json
import datetime
import numpy as np
from pprint import pprint

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import CARs_volume as cars
# normalize_joint_center, scale_points_to_surface, calc_spherical_surface_area, plot_on_sphere, draw_all_points_on_sphere, add_nose_and_other_gh, find_centroid, convert_cartesian_to_latlon_rad, convert_latlon_to_cartesian


def new_process_CARs_vid_from_file(filepath, pose_landmarker):
    real_landmarks = []
    with pose_landmarker as landmarker:
        cap = cv2.VideoCapture(vid_path)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        while True:
            success, img = cap.read()  # this gives one frame at a time
            if not success:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=img)
            timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            # timestamp MUST be an int :)
            # the detect_for_video returns two arrays:
            # >> pose_landmarks: each is "Normalized Landmark represents a point in 3D space with x, y, z coordinates. x and y are normalized to [0.0, 1.0] by the image width and height respectively. z represents the landmark depth, and the smaller the value the closer the landmark is to the camera. The magnitude of z uses roughly the same scale as x."
            # >> pose_world_landmarks: each is "Landmark represents a point in 3D space with x, y, z coordinates. The landmark coordinates are in meters. z represents the landmark depth, and the smaller the value the closer the world landmark is to the camera."
            landmark_result = landmarker.detect_for_video(
                mp_image, timestamp).pose_world_landmarks
            real_landmarks.append(landmark_result)
    return real_landmarks


def new_normalize_joint_center(real_landmarks, target_joint_id, moving_joint_id):
    tj_path = []
    mj_path = []
    for lm_list in real_landmarks:
        targ = lm_list[0]
        # tj == a Landmark object
        tj = targ[target_joint_id]
        mj = targ[moving_joint_id]
        # TODO filter out low-visibility or low-presence scores?
        if mj.presence < .9 or mj.visibility < .9:
            # filtering out some lower values...
            continue
        tj_coords = [tj.x, tj.y, tj.z]
        mj_coords = [mj.x, mj.y, mj.z]
        tj_path.append(tj_coords)
        mj_path.append(mj_coords)
    # convert to np.array for quick vector handling
    tj_array = np.array(tj_path)
    mj_array = np.array(mj_path)
    idealized_joint_center = tj_array[0]
    final_mj_path = np.zeros((len(mj_array), 3))
    radii = []
    for i, pt in enumerate(tj_array):
        diff = pt - idealized_joint_center
        altered_mj_pt = mj_array[i] + diff
        # radius = np.linalg.norm(altered_mj_pt - idealized_joint_center)
        radius = np.linalg.norm(mj_array[i] - tj_array[i])
        radii.append(radius)
        # print(f"Altering mj point:\n prev: {mj_array[i]}\n new: {altered_mj_pt}")
        final_mj_path[i] = altered_mj_pt
    avg_radius = sum(radii) / len(mj_array)
    median_radius = sorted(radii)[len(mj_array)//2]
    longest_radius = sorted(radii)[-1]

    return avg_radius, idealized_joint_center, final_mj_path


# -------------------
# TODO single image mp run (hand in per frame from cv2?) --> basically just use "landmarker.detect(frame)" instead of "landmarker.detect_for_video" ; it's slower but maybe better?


# this is the seed of an app that:
# 1. processes a video into estimated joint positions
# - need to refactor using new mp flow ()
# DONE
# 2. normalizes and scales the points of a moving joint around a target joint to capture...
# 3. calculates the surface area defined by the moving_joint_path, which gives a ratio of 'reachable space'
# 4. as well as a 'summary' of zones entered into
# FUTURE:
# - ability to analyze particular zone of CAR
# - historical and data-analysis of a persons filmed CARs
# HEADS UP: this is the new (Sept '23) version, incorporating mediapipe update (https://developers.google.com/mediapipe/solutions/guide#legacy)

# run serializer
def save_run_to_json(landmark_array):
    date_string = datetime.datetime.today().strftime("%d_%m_%Y__%H:%M:%S")
    file_string = f'/Users/williamhbelew/Hacking/ocv_playground/CARs_app/lm_runs_json/sample_landmarks_{date_string}.json'
    with open(file_string, 'w') as landmark_json:
        to_json = {}
        for i, pose in enumerate(lm_array):
            to_json[i] = {}
            targ_pose = to_json[i]
            for j, joint in enumerate(pose[0]):
                # each joint is a Landmark() object
                targ_pose[j] = {}
                targ_joint = targ_pose[j]
                targ_joint["x"] = joint.x
                targ_joint["y"] = joint.y
                targ_joint["z"] = joint.z
                targ_joint["visibility"] = joint.visibility
                targ_joint["presence"] = joint.presence

        json.dump(to_json, landmark_json)

# TODO run deserializer (back into Landmark obj) --> can I just pickle?


if __name__ == "__main__":
    # video mp run!
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    mp_pose_model_heavy_path = '/Users/williamhbelew/Hacking/ocv_playground/CARs_app/models/pose_landmarker_heavy.task'

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=mp_pose_model_heavy_path),
        running_mode=VisionRunningMode.VIDEO
        # OTHER config options:
        # num_poses
        # min_pose_detection_confidence
        # min_pose_presence_confidence
        # min_tracking_confidence
        # output_segmentation_masks
        # result_callback --> livestream mode only
    )

    # get vid filepath
    vid_path = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/sample_CARs/R_gh_bare_output.mp4"
    # construct landmarker to use (w/ correct options --> see above)
    pose_landmarker = PoseLandmarker.create_from_options(options)
    # build lm_array w/ cv2 > mediapipe.landmarker
    tStart = time.time()
    lm_array = new_process_CARs_vid_from_file(vid_path, pose_landmarker)
    tEnd = time.time()
    landmarking_time = tEnd - tStart
    # each run, save to it's own json for easy unpacking/viewing
    # save_run_to_json(lm_array)

    # normalize points around target joint
    # TODO determine WHICH joint is moving most to determine which CAR, side, etc it is
    # TODO update normalize_joint_center to work with NEW version of landmark array
    avg_radius, jt_center, mj_path_array = new_normalize_joint_center(
        lm_array, 12, 14)
    # TODO rewrite the plotting functions to handle the new shapes available (no gloabl refs!)
    surface_area = cars.draw_all_points_on_sphere(
        avg_radius, jt_center, mj_path_array)
    print(surface_area, landmarking_time)
