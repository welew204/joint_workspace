import cv2
import mediapipe as mp
import numpy as np
import poseDetectionModule as pm

# work through flow, refactor into version that slowly builds a cumulative dictionary of points
# test is: can I easily use outputs to work new validation tests

# INGEST VIDEO (manual trimming)
# current version
# OR mobile-shot video

# process video into dictionary
# save to json for easier pickup
# (replace with pickle process)


def get_landmarker_options(pose_model_path):
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
    return options


def process_CARs_from_video(video_filepath, pose_landmarker):
    """ INPUT: full path to video file, a mp.PoseLandmarker object, built with correct options """
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
            # TODO test that this unpacking works as expected....
            normalized_landmark_result, real_landmark_result = landmarker.detect_for_video(
                mp_image, timestamp)
            landmarks_dict_by_frame[frame] = {
                "timestamp": timestamp,
                "real_landmark_result": real_landmark_result,
                "normalized_landmark_result": normalized_landmark_result}
            frame += 1
    return landmarks_dict_by_frame
# build from json/pickle

# calc avg_radius, jt_center, mj_path_array

# smooth points
# calc centroid
# sort points (use x-axis of joint as starting point)

# partition by displacement
