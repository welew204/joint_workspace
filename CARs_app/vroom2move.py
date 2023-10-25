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
        cap = cv2.VideoCapture(filepath)
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


def smooth_landmarks(surfaced_mj_path, window_size=5):
    # print(surfaced_mj_path)
    output_array = []
    l = 0 - (window_size // 2)
    r = l + (window_size - 1)
    end_r = r
    while l < len(surfaced_mj_path - (window_size // 2)):
        if l < 0:
            neg_window = surfaced_mj_path[l:]
            pos_window = surfaced_mj_path[:r]
            window = np.concatenate((neg_window, pos_window))
        else:
            window = surfaced_mj_path[l:r]
        avg_vect = cars.find_centroid(window)
        output_array.append(avg_vect)

        if r == len(surfaced_mj_path):
            r = -1
        elif r > len(surfaced_mj_path) + end_r:
            break
        r += 1
        l += 1
    return np.array(output_array)


def new_normalize_joint_center(real_landmarks, target_joint_id, moving_joint_id, tight_tolerance=False, thin_points=False):
    tj_path = []
    mj_path = []
    for i, lm_list in enumerate(real_landmarks):
        if thin_points:
            if i % 3 == 0:
                continue
        targ = lm_list[0]
        # tj == a Landmark object
        tj = targ[target_joint_id]
        mj = targ[moving_joint_id]
        tolerance_val = .8  # for weeding out low-visibility or low-presence scores
        if type(mj) == dict:
            # handling deserialized json obj, rather than og Landmark objects
            if tight_tolerance:
                if mj["presence"] < tolerance_val or mj["visibility"] < tolerance_val:
                    # filtering out some lower values...
                    print(f"filtering out mj from frame {i}")
                    pprint(mj)
                    continue
            tj_coords = [tj["x"], tj["y"], tj["z"]]
            mj_coords = [mj["x"], mj["y"], mj["z"]]
        else:
            # handling Landmark objects
            if tight_tolerance:
                if mj.presence < tolerance_val or mj.visibility < tolerance_val:
                    # filtering out some lower values...
                    continue
            tj_coords = [tj.x, tj.y, tj.z]
            mj_coords = [mj.x, mj.y, mj.z]
        tj_path.append(tj_coords)
        mj_path.append(mj_coords)
    # convert to np.array for quick vector handling
    tj_array = np.array(tj_path)
    mj_array = np.array(mj_path)
    prev_idealized_joint_center = tj_array[0]
    idealized_joint_center = cars.find_centroid(tj_array)
    final_mj_path = np.zeros((len(mj_array), 3))
    radii = []
    for i, pt in enumerate(tj_array):
        # TODO do a difference based on each frame, not on a single position
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
    # using LONGEST bc the thinking is that when the arm is longest (across the screen, it will also be most accurate)

    return longest_radius, idealized_joint_center, final_mj_path


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
        for i, pose in enumerate(landmark_array):
            if pose == []:
                print(f"frame {i} was empty!")
                continue
            to_json[i] = {}
            targ_pose = to_json[i]
            for j, joint in enumerate(pose[0]):
                # at some point the POSE is empty? i == 146

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


def run_from_json(json_path):
    with open(json_path, 'r') as landmark_json:
        lmsjson = json.load(landmark_json)
        lm_result = []
        for k, v in lmsjson.items():
            joint_keys = [int(i) for i in v.keys()]
            joint_keys.sort()
            joints = [v[str(jkey)] for jkey in joint_keys]
            lm_result.append([joints])
        # pprint(lm_result[:5])
        return lm_result


if __name__ == "__main__":

    running_from_vid_file = False

    # video mp run!
    if running_from_vid_file:
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        mp_pose_model_heavy_path = '/Users/williamhbelew/Hacking/ocv_playground/CARs_app/models/pose_landmarker_heavy.task'

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=mp_pose_model_heavy_path),
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
        vid_path_R_gh = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/sample_CARs/R_gh_bare_output.mp4"
        vid_path_R_gh_small = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/sample_CARs/R_GH_small_output.mp4"
        vid_path_R_gh_side = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/sample_CARs/R_GH_side_output.mp4"
        vid_path_R_gh_front_small = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/sample_CARs/R_GH_front_small_output.mp4"
        vid_path_R_hip_side_quad = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/sample_CARs/R_hip_quadruped_output.mp4"
        vid_path_R_hip_side_quad_small = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/sample_CARs/R_hip_quadruped_small_output.mp4"
        # construct landmarker to use (w/ correct options --> see above)
        pose_landmarker = PoseLandmarker.create_from_options(options)
        # build lm_array w/ cv2 > mediapipe.landmarker
        tStart = time.time()
        lm_array = new_process_CARs_vid_from_file(
            vid_path_R_hip_side_quad_small, pose_landmarker)

        tEnd = time.time()
        landmarking_time = tEnd - tStart
        print("That took... ", landmarking_time)
        # each run, save to it's own json for easy unpacking/viewing
        save_run_to_json(lm_array)
        exit()
    else:
        json_file_R_GH_path = '/Users/williamhbelew/Hacking/ocv_playground/CARs_app/lm_runs_json/sample_landmarks_26_09_2023__11:33:05.json'
        json_file_R_GH_path_small_side = '/Users/williamhbelew/Hacking/ocv_playground/CARs_app/lm_runs_json/sample_landmarks_27_09_2023__small_R_GH.json'
        json_file_R_GH_path_side = '/Users/williamhbelew/Hacking/ocv_playground/CARs_app/lm_runs_json/sample_landmarks_27_09_2023__side_R_GH.json'
        json_file_R_GH_path_front_small = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/lm_runs_json/sample_landmarks_04_10_2023__front_R_gh_small.json"
        json_file_R_hip_path_quad_side = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/lm_runs_json/sample_landmarks_04_10_2023__side_R_hip_quad.json"
        json_file_R_hip_path_quad_side_small = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/lm_runs_json/sample_landmarks_04_10_2023__side_R_hip_quad_small.json"

        lm_array = run_from_json(json_file_R_GH_path)

    # normalize points around target joint
    # TODO determine WHICH joint is moving most to determine which CAR, side, etc it is

    # R gh == 12, R elbow = 14
    # L gh == 11, L elbow = 13
    # R hip == 24, R knee = 26
    avg_radius, jt_center, mj_path_array = new_normalize_joint_center(
        lm_array, 12, 14, tight_tolerance=False, thin_points=False)

    """ arr = np.array([[3, 5, 7],
                    [2, 4, 6],
                    [5, 7, 8],
                    [7, 2, 3],
                    [19, 4, -1],
                    [12, 12, 1],
                    [32, 45, 3]])
    print(mj_path_array[0])
    print(mj_path_array[-2:0])
    print(mj_path_array[-2:])
    arr2 = arr[0:3]
    arr3 = arr[1:4]
    arr4 = arr[2:5]
    print(arr2)
    exit() """

    smoothd_lms = smooth_landmarks(mj_path_array)
    # cars.draw_all_points_on_sphere(
    #    avg_radius, jt_center, mj_path_array, lm_array, scale_to_sphere=True)
    cars.draw_all_points_on_sphere(
        avg_radius, jt_center, smoothd_lms, lm_array, scale_to_sphere=True)
    exit()
    """ PREP'D some json for RKB 
    with open('/Users/williamhbelew/Hacking/ocv_playground/CARs_app/lm_runs_json/sample_landmarks_normalized_GH.json', 'w') as landmark_json:
        to_json = {}
        for i, pose in enumerate(mj_path_array):

            to_json[i] = {}

            to_json[i]["x"] = pose[0]
            to_json[i]["y"] = pose[1]
            to_json[i]["z"] = pose[2]

        json.dump(to_json, landmark_json) """

    # print(landmarking_time)
