import cv2
import mediapipe as mp
import numpy as np
import poseDetectionModule as pm
import datetime
import json
import vroom2move as v2m

# work through flow, refactor into version that slowly builds a cumulative dictionary of points
# test is: can I easily use outputs to work new validation tests

# INGEST VIDEO (manual trimming)
# current version
# OR mobile-shot video

# process video into dictionary
# save to json for easier pickup


def get_landmarker_with_options(pose_model_path):
    """INPUT: filepath to model being used\n
    OUTPUT: PoseLandmarker object, with options static video options passed in"""
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=pose_model_path),
        running_mode=VisionRunningMode.VIDEO
        # OTHER possible config options:
        # num_poses
        # min_pose_detection_confidence
        # min_pose_presence_confidence
        # min_tracking_confidence
        # output_segmentation_masks
        # result_callback --> livestream mode only
    )
    pose_landmarker = PoseLandmarker.create_from_options(options)
    return pose_landmarker


def process_CARs_from_video(video_filepath, pose_landmarker):
    """ INPUT: full path to video file, a mp.PoseLandmarker object, built with correct options\n
    OUTPUT: dictionary with each frame keying a dictionary with (real_landmarks, normalized_landmarks (0.0-1.0), and timestamp)"""
    landmarks_dict_by_frame = {}
    with pose_landmarker as landmarker:
        cap = cv2.VideoCapture(video_filepath)
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame = 0
        while True:
            success, img = cap.read()  # this gives one frame at a time
            if not success:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=img)
            timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            # timestamp (ms) MUST be an int :)
            # the detect_for_video returns two arrays:
            # >> pose_landmarks: each is "Normalized Landmark represents a point in 3D space with x, y, z coordinates. x and y are normalized to [0.0, 1.0] by the image width and height respectively. z represents the landmark depth, and the smaller the value the closer the landmark is to the camera. The magnitude of z uses roughly the same scale as x."
            # >> pose_world_landmarks: each is "Landmark represents a point in 3D space with x, y, z coordinates. The landmark coordinates are in meters. z represents the landmark depth, and the smaller the value the closer the world landmark is to the camera."

            landmarker_result = landmarker.detect_for_video(
                mp_image, timestamp)

            landmarks_dict_by_frame[frame] = {
                "timestamp": timestamp,
                "real_landmark_result": landmarker_result.pose_world_landmarks,
                "normalized_landmark_result": landmarker_result.pose_landmarks}
            frame += 1
    return landmarks_dict_by_frame


def serialize_to_json(landmark_dict, json_string_id):
    """INPUT: dictionary, json string identifier that will get added to filepath\n
    OUTPUT: json file (dumped dictionary)"""
    date_string = datetime.datetime.today().strftime("%d-%m-%y_%H%M%S")
    file_string = f"/Users/williamhbelew/Hacking/ocv_playground/CARs_app/json_lm_store/landmarks__{json_string_id}__{date_string}.json"
    with open(file_string, 'w') as landmark_json:
        # TODO any filtering/processing to slim the dict?
        json.dump(landmark_dict, landmark_json)

# build from json


def run_from_json(json_path):
    """INPUT: json filepath\n
    OUTPUT: json-loaded dictionary"""
    with open(json_path, 'r') as landmark_json:
        lmsjson = json.load(landmark_json)
        return lmsjson

# calc avg_radius, jt_center, mj_path_array

# TODO test this shnizzz!


def process_landmarks(landmark_dict, target_joint_id, moving_joint_id, tight_tolerance=False, thin_points=False):
    """INPUT: landmark dictionary (will get added to, and CHANGED), target and moving joint id's, and flag-options for normalizing ("tight_tolerance", "thin_points"\n
    OUTPUT: json file (dumped dictionary)"""
    real_landmarks = [[frame, lm["real_landmark_result"]]
                      for frame, lm in landmark_dict]
    # ensure that the landmarks in frame_order
    real_landmarks.sort(lambda i: i[0])
    real_landmarks = [i[1] for i in real_landmarks]
    avg_radius, jt_center, mj_path_array = v2m.new_normalize_joint_center(
        real_landmarks, target_joint_id, moving_joint_id, tight_tolerance=tight_tolerance, thin_points=thin_points)
    # adding a single mj_position per frame
    for frame, normalized_mj_position in enumerate(mj_path_array):
        landmark_dict[frame]["norm_mj_position"] = normalized_mj_position
    landmark_dict["avg_radius"] = avg_radius
    landmark_dict["jt_center"] = jt_center
    landmark_dict["tj_mj_ids"] = [target_joint_id, moving_joint_id]
    return landmark_dict


# smooth points
# calc centroid
# sort points (use x-axis of joint as starting point)
# partition by displacement


# FOR TESTING
if __name__ == "__main__":
    pose_landmarker = get_landmarker_with_options(
        "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/models/pose_landmarker_heavy.task")
    vid_path_R_gh = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/sample_CARs/R_gh_bare_output.mp4"
    test_result = process_CARs_from_video(vid_path_R_gh, pose_landmarker)
    exit()
