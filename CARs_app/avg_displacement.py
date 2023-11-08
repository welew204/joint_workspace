import CARs_volume as cars
import vroom2move as v2m

import numpy as np
from pprint import pprint


# -----previous idea
# *conical spherical volume* == the spherical cap PLUS the cone-volume underneath it (ie a depleted ice cream cone)

# per point:
# calc the elev (down from centroid) and the azimuth (rotation ABOUT)
# calc the diff in azimuth from previous point; calc the proportion of the circle this, and the sign
# use this proportion, the elevation, & radius to calculate the *conical spherical volume* of:
# the previous point, the current point => return: the mean of these two
# increment this in the total conical volume
# sidebar process per zone (if IN a certain zone, then assign to that zonal tally?)


def calc_reachable_area(sorted_mj_points, centroid):
    total_reach = 0
    for i,  pt in enumerate(sorted_mj_points):
        reach = np.linalg.norm(pt - centroid)
        # next, want to calc the area of a triangle (speherical triangle?) if I use neighboring points
        # also, re-form partition function as a sorting function that id's what zone a point is in, to tally avgs (and eventually SAs) by per zone
        total_reach += reach
    return total_reach // len(sorted_mj_points)
# ----------

# -----executed idea (below):

# take in an array of normalized and scaled points (same radius)

# calculate centroid of all points, ie: the central axis for all calcs
# sort points into order of rotation from initial point
# calc the dispalcement (aka how far the point is from the centroid)
# partition into near/median/far points
# calc the avg of those (furthest) points


def sort_points_by_angle(mj_cart_pts, centroid):
    """Input smoothed array of points, and centroid (of ENTIRE path); 
    Outputs sorted array of moving joint points in order of rotation from intial point of array (so modify this if working zonally)"""
    origin_rotational_angle = np.empty((3,))
    output_array = []
    riks_array = []
    for i, pt in enumerate(mj_cart_pts):
        vect_from_centroid = centroid - pt
        if i == 0:
            origin_rotational_angle = vect_from_centroid

        # cos = cos_between(origin_rotational_angle, vect_from_centroid)
        # arccos = arccos_between(origin_rotational_angle, vect_from_centroid)
        theta = angle_between(origin_rotational_angle, vect_from_centroid)
        arctan_theta = arctan2_angle_between(
            vect_from_centroid, origin_rotational_angle)
        riks_array.append([i, theta, arctan_theta])
        # TODO clean up this Python into numpy, think 'broadcasting'
        output_array.append([pt[0], pt[1], pt[2], arctan_theta])
    output_array.sort(key=(lambda x: x[3]))
    return output_array


def partition_by_displacement(sorted_mj_array, centroid, angular_span=360, window_size=3):
    '''INPUT: window size is in degrees; array is in form of ((x,y,z,theta))
    OUTPUT FORMAT of elements: [min_norm, Max_norm, n_of_points, (x,y,z)], the coord is of the max'''

    # init the containers for the closest, middle, and furthest 'islands' away from home base
    number_of_windows = int(angular_span // window_size)
    # [min_norm, Max_norm, n_of_points, (x,y,z)], the coord is of the max
    buckets = [[] for _ in range(number_of_windows)]
    output = [[0, 0, 0, (0, 0, 0)] for _ in range(number_of_windows)]

    # partition by angle instead of by points
    starting_theta = sorted_mj_array[0][3]
    for idx, pt in enumerate(sorted_mj_array):
        theta = pt[3] - starting_theta
        bucket = int(theta // window_size)
        buckets[bucket].append(np.array(pt[:3]))
    print()
    for i, sample in enumerate(buckets):
        if len(sample) == 0:
            # case: no points in this angular range
            continue
        sample_by_displacement = []
        for point in sample:
            displacement = np.linalg.norm(point - centroid)
            out = np.array([point[0], point[1], point[2], displacement])
            sample_by_displacement.append(out)
        sample_by_displacement = np.array(sample_by_displacement)
        min_index = np.argmin(sample_by_displacement[:, 3])
        max_index = np.argmax(sample_by_displacement[:, 3])

        # FORMAT of output elements: [min_norm, Max_norm, n_of_points, (x,y,z)], the coord is of the max
        output[i] = [sample_by_displacement[min_index][3], sample_by_displacement[max_index][3], len(
            sample_by_displacement), sample_by_displacement[max_index][:3]]

    return output


def avg_displacement(displacements):
    '''IN: np.array() of displacement for each max_point'''
    total = np.sum(displacements)
    no_of_elements = len(displacements)
    return total/no_of_elements


def cos_between(v1, v2):
    #    Compute the dot product
    dot_product = np.dot(v1, v2)

    # Compute the norms of the vectors
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # Calculate cosine theta
    cos_theta = dot_product / (norm_v1 * norm_v2)

    # Ensure the value lies between -1 and 1 to avoid numerical issues
    cos_theta = np.clip(cos_theta, -1, 1)
    return cos_theta


def arccos_between(v1, v2):
    # Compute the dot product
    dot_product = np.dot(v1, v2)

    # Compute the norms of the vectors
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # Calculate cosine theta
    cos_theta = dot_product / (norm_v1 * norm_v2)

    # Ensure the value lies between -1 and 1 to avoid numerical issues
    cos_theta = np.clip(cos_theta, -1, 1)

    # Calculate the angle in radians
    theta = np.arccos(cos_theta)

    return theta


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


def angle_between(v1, v2):
    # compute arctan2 to get angle in radians, then convert
    # wathc out for negative angle, so in all cases add 2pi then convert to degrees
    # Compute the dot product
    dot_product = np.dot(v1, v2)

    # Compute the norms of the vectors
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    # Calculate cosine theta
    cos_theta = dot_product / (norm_v1 * norm_v2)

    # Ensure the value lies between -1 and 1 to avoid numerical issues
    cos_theta = np.clip(cos_theta, -1, 1)

    # Calculate the angle in radians
    theta = np.arccos(cos_theta)

    # to handle change in (+ / -) of

    # Convert to degrees for a more intuitive result
    angle_deg = np.degrees(theta)

    return angle_deg


def full_flow(json_path, title_of_run, target_joint_id, moving_joint_id, draw=False):
    """the full workflow from points (json) to avg_displacement
    target_joint -> joint being assessed; moving_joint -> path being evaluated
    R gh == 12, R elbow = 14
    L gh == 11, L elbow = 13
    R hip == 24, R knee = 26
    """
    landmarks = v2m.run_from_json(json_path)

    avg_radius, jt_center, mj_path_array = v2m.new_normalize_joint_center(
        landmarks, target_joint_id, moving_joint_id, tight_tolerance=False, thin_points=False)

    smoothed_mvj_points = v2m.smooth_landmarks(mj_path_array)
    centroid = cars.find_centroid(smoothed_mvj_points)

    sorted_smoothed_mvj_points = sort_points_by_angle(
        smoothed_mvj_points, centroid)

    output = partition_by_displacement(
        sorted_smoothed_mvj_points, centroid)
    displacements = [bucket[1] for bucket in output if bucket[1] != 0]

    print(
        f"avg displacement of closest points for {title_of_run}\n--->{avg_displacement(np.array(displacements))}")
    if draw:
        max_points = [np.array(bucket[3]) for bucket in output]
        cars.draw_all_points_on_sphere(avg_radius, jt_center, np.array(
            max_points), landmarks, scale_to_sphere=True)


if __name__ == "__main__":
    json_file_R_GH_path = '/Users/williamhbelew/Hacking/ocv_playground/CARs_app/lm_runs_json/sample_landmarks_26_09_2023__11:33:05.json'
    json_file_R_GH_path_small_side = '/Users/williamhbelew/Hacking/ocv_playground/CARs_app/lm_runs_json/sample_landmarks_27_09_2023__small_R_GH.json'
    json_file_R_GH_path_side = '/Users/williamhbelew/Hacking/ocv_playground/CARs_app/lm_runs_json/sample_landmarks_27_09_2023__side_R_GH.json'
    json_file_R_GH_path_front_small = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/lm_runs_json/sample_landmarks_04_10_2023__front_R_gh_small.json"
    json_file_R_hip_path_quad_side = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/lm_runs_json/sample_landmarks_04_10_2023__side_R_hip_quad.json"
    json_file_R_hip_path_quad_side_small = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/lm_runs_json/sample_landmarks_04_10_2023__side_R_hip_quad_small.json"
    json_file_L_wrist_front_full = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/lm_runs_json/sample_landmarks_01_11_2023__16:06:32__L_wrist_front_full.json"
    json_file_L_wrist_oblique_full = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/lm_runs_json/sample_landmarks_01_11_2023__16:13:03__L_wrist_oblique_full.json"
    json_file_L_wrist_oblique_small = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/lm_runs_json/sample_landmarks_01_11_2023__16:14:38__L_wrist_oblique_small.json"
    json_file_L_wrist_oblique_full_zoomed_out = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/lm_runs_json/sample_landmarks_01_11_2023__16:22:46__L_wrist_full_oblique_zoomed_out.json"
    json_file_L_ankle_front_full = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/lm_runs_json/sample_landmarks_01_11_2023__16:43:28__L_ankle_front_full.json"
    json_file_L_ankle_front_small = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/lm_runs_json/sample_landmarks_01_11_2023__16:45:04__L_ankle_front_small.json"
    json_file_L_ankle_side_full = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/lm_runs_json/sample_landmarks_01_11_2023__16:46:22__L_ankle_side_full.json"
    json_file_L_ankle_side_small = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/lm_runs_json/sample_landmarks_01_11_2023__16:47:27__L_ankle_side_small.json"
    json_file_L_hip_standing_oblique_full = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/lm_runs_json/sample_landmarks_02_11_2023__13:47:01__L_hip_standing_oblique_full.json"
    json_file_L_hip_standing_oblique_small = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/lm_runs_json/sample_landmarks_02_11_2023__13:51:03__L_hip_standing_oblique_small.json"

    # R gh == 12, R elbow = 14
    # L gh == 11, L elbow = 13
    # R hip == 24, R knee = 26
    # L hip == 23, L knee = 25
    # L wrist == 15, L index-tarsal == 19
    # L ankle == 27, L hallux == 31
    # full_flow(json_file_R_GH_path, "R GH - front", 12, 14)
    # full_flow(json_file_R_GH_path_front_small, "R GH - front, small", 12, 14)
    # full_flow(json_file_R_GH_path_side, "R GH - side", 12, 14)
    # full_flow(json_file_R_GH_path_small_side,"R GH - side, small", 12, 14)
    full_flow(json_file_L_hip_standing_oblique_full,
              "L hip - side, full", 23, 25, draw=True)
    full_flow(json_file_L_hip_standing_oblique_small,
              "L hip - side, small", 23, 25, draw=True)
    exit()

    landmarks = v2m.run_from_json(json_file_R_GH_path)
    avg_radius, jt_center, mj_path_array = v2m.new_normalize_joint_center(
        landmarks, 12, 14, tight_tolerance=False, thin_points=False)
    smoothed_mvj_points = v2m.smooth_landmarks(mj_path_array)
    centroid = cars.find_centroid(smoothed_mvj_points)
    sorted_smoothed_mvj_points = sort_points_by_angle(
        smoothed_mvj_points, centroid)
    ss_mvj_path_no_theta = [pt[:3] for pt in sorted_smoothed_mvj_points]
    ss_mvj_path_by_quadrant = cars.partition_mj_path(
        jt_center, ss_mvj_path_no_theta)
    for zone, pts in ss_mvj_path_by_quadrant.items():
        sorted_zonal_points = sort_points_by_angle(pts, centroid)
        # calc'ing the angular span for this zone of points
        spanning_angle_about_rotation = arctan2_angle_between(
            centroid - sorted_zonal_points[0][:3], centroid - sorted_zonal_points[-1][:3])
        output = partition_by_displacement(
            sorted_zonal_points, centroid, angular_span=spanning_angle_about_rotation)
        displacements = [bucket[1] for bucket in output if bucket[1] != 0]
        print(
            f"avg displacement of closest points for zone #{zone}\n--->{avg_displacement(np.array(displacements))}")
    exit()
