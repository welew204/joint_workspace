import cv2
import mediapipe as mp
import numpy as np
import poseDetectionModule as pm
import datetime
import json
import vroom2move as v2m
import CARs_volume as cars
import avg_displacement as avd
import collections

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
    OUTPUT: dumps dict to json file (returns filepath of resulting json)"""
    date_string = datetime.datetime.today().strftime("%d-%m-%y_%H%M%S")
    file_string = f"/Users/williamhbelew/Hacking/ocv_playground/CARs_app/json_lm_store/landmarks__{json_string_id}__{date_string}.json"
    with open(file_string, 'w') as landmark_json:
        # TODO any filtering/processing to slim the dict?

        real_lm_to_json = v2m.create_json([v['real_landmark_result']
                                           for k, v in landmark_dict.items()])
        normalized_lm_to_json = v2m.create_json([v['normalized_landmark_result']
                                                 for k, v in landmark_dict.items()])
        timestamps = [v["timestamp"] for k, v in landmark_dict.items()]
        to_json = {i: {
            "timestamp": timestamps[i],
            "real_landmark_result": real_lm_to_json[i],
            "normalized_landmark_result": normalized_lm_to_json[i]} for i in range(len(timestamps))}
        json.dump(to_json, landmark_json)
    return file_string

# build from json


def run_from_json(json_path):
    """INPUT: json filepath\n
    OUTPUT: json-loaded dictionary (all numeric keys are 'string', not int)"""
    with open(json_path, 'r') as landmark_json:
        lms_from_json = json.load(landmark_json)
        return lms_from_json

# calc avg_radius, jt_center, mj_path_array


# TODO test this shnizzz!
"""like--actually write some fuggin tests bischhhh"""


def process_landmarks(landmark_dict, target_joint_id, moving_joint_id, tight_tolerance=False, thin_points=False):
    """INPUT: landmark dictionary (will get added to, and CHANGED), target and moving joint id's, and flag-options for normalizing ("tight_tolerance", "thin_points"\n
    OUTPUT: landmark dictionary, w/ keys for each frame (string), plus some aggregated vals (mj_path_array, avg_radius, jt_center, tj_mj_ids)"""
    # og_landmark_dict_for_ref = copy.deepcopy(landmark_dict)

    # first get the list of poses (each a dict)
    real_lms = [v['real_landmark_result']
                for k, v in landmark_dict.items()]
    # then, per pose, add list of joint landmarks (shed the keys)
    listOfLists_landmarks = []
    for pose in real_lms:
        pose_array = [v for k, v in pose.items()]
        # putting this list of joint positions INTO another list to match other flow
        # TODO clean this up....
        listOfLists_landmarks.append([pose_array])

    avg_radius, jt_center, mj_path_array = v2m.new_normalize_joint_center(
        listOfLists_landmarks, target_joint_id, moving_joint_id, tight_tolerance=tight_tolerance, thin_points=thin_points)
    # adding a single mj_position per frame
    for frame, normalized_mj_position in enumerate(mj_path_array):
        landmark_dict[str(frame)]["norm_mj_position"] = normalized_mj_position
    landmark_dict["mj_path_array"] = mj_path_array
    landmark_dict["avg_radius"] = avg_radius
    landmark_dict["jt_center"] = jt_center
    landmark_dict["tj_mj_ids"] = [target_joint_id, moving_joint_id]
    return landmark_dict

# smooth points


def add_smoothed_points(landmark_dict, window_size=5):
    """INPUT: landmark_dict, (opt) size of window to use in smoothing the points;
    OUTPUT: updated landmark_dict, with smoothed points added PER frame ('smoothed_coord') and as a discrete array ('smoothed_mj_path')"""
    # TODO filter outliers?

    mj_path_array = landmark_dict["mj_path_array"]
    smoothed_mvj_points = v2m.smooth_landmarks(
        mj_path_array, window_size=window_size)
    # adding each *smoothed* point to the appropriate frame

    # TODO add test to ensure len of smoothed is same as number of frames
    for p, point in enumerate(smoothed_mvj_points):
        landmark_dict[str(p)]["smoothed_coord"] = point
    landmark_dict["smoothed_mj_path"] = smoothed_mvj_points

    return landmark_dict


def determine_quadrant(coord, jt_center, epsilon=.001):
    """INPUT: point in question (3d coord), jt-center (also 3d coord), epsilon (amount to handle near-zero points; make epsilon larger if I want to treat close values as boundary values)
    OUTPUT: which zone (0 - 7), int
    """
    k_to_q = {"000": 0,  # [-,-,-]
              "001": 1,  # [-,-,+]
              "010": 2,  # [-,+,-]
              "011": 3,  # [-,+,+]
              "100": 4,  # [+,-,-]
              "101": 5,  # [+,-,+]
              "110": 6,  # [+,+,-]
              "111": 7}  # [+,+,+]
    q_key = ""
    # old version that was skewing...
    q_key += "1" if coord[0] > (jt_center[0]+epsilon) else "0"
    q_key += "1" if coord[1] > (jt_center[1]+epsilon) else "0"
    q_key += "1" if coord[2] > (jt_center[2]+epsilon) else "0"
    return k_to_q[q_key]

# calc centroid


def add_centroid(landmark_dict):
    centroid = cars.find_centroid(landmark_dict["smoothed_mj_path"])
    landmark_dict["centroid"] = centroid
    return landmark_dict

# from this SO: https://stackoverflow.com/a/56401672/19589299
# and described further here: http://johnblackburne.blogspot.com/2012/05/angle-between-two-3d-vectors.html


def arctan2_angle_between(v1, v2):
    arg1 = np.cross(v1, v2)
    arg2 = np.dot(v1, v2)
    # guidance from chatGPT
    arg1_cross = np.linalg.norm(arg1)
    angle = np.arctan2(arg1_cross, arg2)
    if arg1[2] < 0:
        # checking if CROSS product is pointing UP or DOWN (which is erased when calc the norm)
        angle = -angle
    if angle < 0:
        angle += 2*np.pi
    angle = np.degrees(angle)
    return angle

# sort points (use x-axis of joint as starting point)


def sort_by_angle(landmark_dict):
    """INPUT: landmark_dict
    OUTPUT: updated landmark_dict w/ new 'angle_sorted_array'"""
    # TODO LATER ....transform points to normalized position (stash this in output dict!)
    # use horizontal frame (x-axis) as 0deg, sort points based on angle
    mj_cart_path = landmark_dict["smoothed_mj_path"]
    centroid = landmark_dict["centroid"]
    sorted_by_angle = avd.sort_points_by_angle(mj_cart_path, centroid)
    landmark_dict['angle_sorted_array'] = sorted_by_angle

    return landmark_dict


def get_zonal_planar_vector(zone, landmark_dict):
    z_to_edge = {0: [-1, -1, 0],
                 1: [0, -1, 1],
                 2: [-1, 1, 0],
                 3: [0, 1, 1],
                 4: [0, -1, -1],
                 5: [1, -1, 0],
                 6: [0, 1, -1],
                 7: [1, 1, 0]}
    planar_vector = np.array(
        z_to_edge[zone])*landmark_dict["avg_radius"] + landmark_dict["jt_center"]
    return planar_vector


def get_planar_crossing(vct_a, vct_b, jt_center):
    k_to_normal = {
        "100": [0, 1, 0],  # aka crossing the xz axis
        "010": [1, 0, 0],  # aka crossing the yz axis
        "001": [0, 0, 1]}  # aka crossing the xy axis
    q_key = ""
    # determine which plane is crossed
    q_key += "1" if min(vct_a[0], vct_b[0]
                        ) < jt_center[0] < max(vct_a[0], vct_b[0]) else "0"
    q_key += "1" if min(vct_a[1], vct_b[1]
                        ) < jt_center[1] < max(vct_a[1], vct_b[1]) else "0"
    q_key += "1" if min(vct_a[2], vct_b[2]
                        ) < jt_center[2] < max(vct_a[2], vct_b[2]) else "0"
    # assert that there's only 1 one in the q_key (aka, only crossing one axis),
    # crossing two is ok, but should be coerced into ONE crossing or the other (basically, pick)
    axis_norm = k_to_normal[q_key]

    # finding intersection of a line with a plane
    # from SO: https://stackoverflow.com/a/18543221
    diff = vct_a - vct_b
    dot = np.dot(np.array(axis_norm), diff)
    # jt_center here is acting as a point on the plane, since it is a point on ANY axial plane
    w = vct_a - jt_center
    factor = -np.dot(axis_norm, w) / dot
    u = diff * factor
    return vct_a + u


def partition_into_zones(landmark_dict):
    """INPUT: landmark_dict
    OUTPUT: add 'zonal_dict' to landmark_dict with an array of smoothed/ang_sorted points in each zone 1-7"""
    jt_center = landmark_dict['jt_center']
    sorted_smoothed_points = [elem[:3]
                              for elem in landmark_dict['angle_sorted_array']]
    zonal_dict = collections.defaultdict(list)
    i = 0
    while i < len(sorted_smoothed_points):
        pt = sorted_smoothed_points[i]
        # there SHOULDn't be back/forth over the axis since I have sorted by angle!

        # calc the intersection of interpolating vector with axis plane, stash this in seperate dict keyed on zone
        zone = determine_quadrant(pt, jt_center)
        if i == 0:
            prev_zone = zone
        elif prev_zone != zone:
            # we have a boundary cross!
            diff_vector = sorted_smoothed_points[i] - \
                sorted_smoothed_points[i-1]

        zonal_dict[zone].append(pt)
        i += 1

    # TODO go thru each zone, sort by angle between axis (which one?) and point
    # use something like the below to map a zone to a correctly oriented unit-vector,
    # then use that comparison vector to scale to a vector of radius-r, then find new vector (angle_comparison_vector) connecting to centroid
    # then use THAT new vector to find angle diff, and sort based on angle diff
    for z in zonal_dict:
        # trying to sort ALL points in a given zone by their angle away from a planar 'edge'
        planar_vector = get_zonal_planar_vector(z, landmark_dict)
        zonal_output_array = []
        for i, pt in enumerate(zonal_dict[z]):
            ang_between = arctan2_angle_between(planar_vector, pt)
            ang_between = ang_between if ang_between < 90 else 360.0-ang_between
            # ang_between2 = arctan2_angle_between(pt, planar_vector)
            # .....> TOTAL validates that the two angles always together equal ~360
            # total = ang_between + ang_between2
            zonal_output_array.append([pt[0], pt[1], pt[2], ang_between])
        zonal_output_array.sort(key=(lambda x: x[3]))

        zonal_dict[z] = zonal_output_array

    landmark_dict["zonal_dict"] = zonal_dict
    return landmark_dict

# partition by displacement


def partition_into_angular_buckets(landmark_dict, angular_window_size=3, by_zone=False):
    """INPUT: landmark_dict, window_size to consider for partitioning 
    OUTPUT: landmark_dict updated with FULL path partition and BY_ZONE partition"""
    # scan through sorted/smoothed points and calc max/min, n-of-points
    centroid = landmark_dict['centroid']
    full_path_output = avd.partition_by_displacement(
        landmark_dict['angle_sorted_array'], centroid, window_size=angular_window_size)
    if by_zone:
        # currently bugging out, want to talk through approach here w/ RKB
        displacement_by_zone = {}
        for z in landmark_dict['zonal_dict']:
            ang_sorted_points_per_zone = landmark_dict['zonal_dict'][z]
            # PREVIOUS, trying to determine angular_span by finding diff between first and last point, but that sorta shrinks the lense of 'zone'
            # angle_span = arctan2_angle_between(ang_sorted_points_per_zone[0][:3], ang_sorted_points_per_zone[-1][:3])
            # angle_span = angle_span if angle_span <= 90.0 else 360 - angle_span
            # perhaps PER zone should be 90deg span...
            angle_span = 90.0
            zonal_output = avd.partition_by_displacement(
                ang_sorted_points_per_zone, centroid, angular_span=angle_span, window_size=angular_window_size)
            displacement_by_zone[z] = zonal_output
        landmark_dict["zonal_displacement_output"] = displacement_by_zone
    landmark_dict["full_path_displacement_output"] = full_path_output

    return landmark_dict


def calc_avg_displacement(displacement_array):
    # use sanitized (sorted/smoothed) points
    return np.sum(displacement_array)/len(displacement_array)


# FOR TESTING
if __name__ == "__main__":
    """ 
    TEST CASE for get_planar_crossing()"""
    test_vect_a = np.array([1, 1, 1])
    test_vect_b = np.array([1, 1, -1])
    planar_intersect = get_planar_crossing(
        test_vect_a, test_vect_b, np.array([0, 0, 0]))
    print(planar_intersect)

    from_vid = False
    if from_vid:
        pose_landmarker = get_landmarker_with_options(
            "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/models/pose_landmarker_heavy.task")
        vid_path_R_gh = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/sample_CARs/R_gh_bare_output.mp4"
        test_result = process_CARs_from_video(vid_path_R_gh, pose_landmarker)
        test_json_filepath = serialize_to_json(
            test_result, "test_result_serialized")
    else:
        test_json_filepath = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/json_lm_store/landmarks__test_result_serialized__13-11-23_070941.json"
    testing_result_dict = run_from_json(test_json_filepath)
    testing_result_dict = process_landmarks(testing_result_dict, 12, 14)
    testing_result_dict = add_smoothed_points(testing_result_dict)
    testing_result_dict = add_centroid(testing_result_dict)
    testing_result_dict = sort_by_angle(testing_result_dict)
    testing_result_dict = partition_into_zones(testing_result_dict)
    testing_result_dict = partition_into_angular_buckets(
        testing_result_dict, by_zone=True)
    displacement_array = [
        elem[1] for elem in testing_result_dict["full_path_displacement_output"] if elem[1] > 0]
    avg_displacement = calc_avg_displacement(displacement_array)
    print(avg_displacement)
    for zone in testing_result_dict["zonal_displacement_output"]:
        zonal_avg_displacement = [
            elem[1] for elem in testing_result_dict["zonal_displacement_output"][zone] if elem[1] > 0]
        print(f"Zone {zone} avg displacment: ",
              calc_avg_displacement(zonal_avg_displacement))

    # R gh == 12, R elbow = 14
    # L gh == 11, L elbow = 13
    # R hip == 24, R knee = 26
    # L hip == 23, L knee = 25
    # L wrist == 15, L index-tarsal == 19
    # L ankle == 27, L hallux == 31

    exit()
