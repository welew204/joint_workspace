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


def sort_points_by_angle(mj_cart_pts):
    """Input smoothed array of points; 
    Outputs sorted array of moving joint points in order of rotation from intial point"""
    centroid = cars.find_centroid(mj_cart_pts)
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
        output_array.append([pt, arctan_theta])
    output_array.sort(key=(lambda x: x[1]))
    return np.array(output_array), centroid


def partition_by_displacement(sorted_mj_array, centroid, window_size=3):
    '''INPUT: window size is in degrees; array is in form of ((x,y,z), theta)
    OUTPUT FORMAT of elements: [min_norm, Max_norm, n_of_points, (x,y,z)], the coord is of the max'''

    # init the containers for the closest, middle, and furthest 'islands' away from home base
    number_of_windows = 360 // window_size
    # [min_norm, Max_norm, n_of_points, (x,y,z)], the coord is of the max
    buckets = [[] for _ in range(number_of_windows)]
    output = [[0, 0, 0, (0, 0, 0)] for _ in range(number_of_windows)]

    # partition by angle instead of by points
    pt_idx = 0
    for idx, pt in enumerate(sorted_mj_array):
        theta = pt[1]
        bucket = int(theta // window_size)
        buckets[bucket].append(pt)
        if bucket > 80:
            pass
    print()
    for i, sample in enumerate(buckets):
        if len(sample) == 0:
            # case: no points in this angular range
            continue
        sample_by_displacement = np.array([
            np.array(point[0], np.linalg.norm(point[0] - centroid)) for point in sample])
        min_index = np.argmin(sample_by_displacement[:, 1])
        max_index = np.argmax(sample_by_displacement[:, 1])

        # FORMAT of output elements: [min_norm, Max_norm, n_of_points, (x,y,z)], the coord is of the max
        output[i] = [sample_by_displacement[min_index][1], sample_by_displacement[max_index][1], len(
            sample_by_displacement), sample_by_displacement[max_index][0]]

    return output


def avg_displacement(pts_with_displacement):
    return np.sum(pts_with_displacement[:, 1])//len(pts_with_displacement)


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


if __name__ == "__main__":
    json_file_R_GH_path = '/Users/williamhbelew/Hacking/ocv_playground/CARs_app/lm_runs_json/sample_landmarks_26_09_2023__11:33:05.json'
    json_file_R_GH_path_small_side = '/Users/williamhbelew/Hacking/ocv_playground/CARs_app/lm_runs_json/sample_landmarks_27_09_2023__small_R_GH.json'
    json_file_R_GH_path_side = '/Users/williamhbelew/Hacking/ocv_playground/CARs_app/lm_runs_json/sample_landmarks_27_09_2023__side_R_GH.json'
    json_file_R_GH_path_front_small = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/lm_runs_json/sample_landmarks_04_10_2023__front_R_gh_small.json"
    json_file_R_hip_path_quad_side = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/lm_runs_json/sample_landmarks_04_10_2023__side_R_hip_quad.json"
    json_file_R_hip_path_quad_side_small = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/lm_runs_json/sample_landmarks_04_10_2023__side_R_hip_quad_small.json"
    landmarks = v2m.run_from_json(json_file_R_GH_path)

    # R gh == 12, R elbow = 14
    # L gh == 11, L elbow = 13
    # R hip == 24, R knee = 26
    avg_radius, jt_center, mj_path_array = v2m.new_normalize_joint_center(
        landmarks, 12, 14, tight_tolerance=False, thin_points=False)

    smoothed_mvj_points = v2m.smooth_landmarks(mj_path_array)

    sorted_smoothed_mvj_points, centroid = sort_points_by_angle(
        smoothed_mvj_points)
    cars.draw_all_points_on_sphere(
        avg_radius, jt_center, np.array(sorted_smoothed_mvj_points[:, 1]), landmarks, scale_to_sphere=True)

    output = partition_by_displacement(
        sorted_smoothed_mvj_points, centroid)

    print("avg displacement of closest points", avg_displacement(output))

    # print("avg displacement of furthest points", avg_displacement(furthest))
