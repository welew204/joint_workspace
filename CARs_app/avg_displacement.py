import CARs_volume as cars
import vroom2move as v2m

import numpy as np
from pprint import pprint

# -----executed idea (below):

# take in an array of normalized and scaled points (same radius)

# calculate centroid of all points, ie: the central axis for all calcs
# sort points into order of rotation from initial point
# calc the dispalcement (aka how far the point is from the centroid)
# partition into near/median/far points
# calc the avg of those (furthest) points

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


def sort_points_by_angle(mj_cart_pts):
    """Input smoothed array of points; 
    Outputs sorted array of moving joint points in order of rotation from intial point"""
    centroid = cars.find_centroid(mj_cart_pts)
    origin_rotational_angle = np.empty((3,))
    output_array = []
    for i, pt in enumerate(mj_cart_pts):
        vect_from_centroid = centroid - pt
        if i == 0:
            origin_rotational_angle = vect_from_centroid
            continue
        theta = angle_between(origin_rotational_angle, pt)
        output_array.append(np.array(pt, theta))
    return output_array.sort(key=(lambda x: x[1])), centroid


def partition_by_displacement(sorted_mj_array, centroid, window_size):
    # init the containers for the closest, middle, and furthest 'islands' away from home base
    nearest = []
    median = []
    furthest = []

    l = 0 - (window_size // 2)
    r = l + (window_size)
    end_r = r
    while True:
        if l < 0:
            neg_window = sorted_mj_array[l:]
            pos_window = sorted_mj_array[:r]
            window = neg_window + pos_window
        elif l > len(sorted_mj_array)-window_size:
            # to handle when we get toward the end of the array,
            # and the r values need to wrap back to the front of the array
            neg_window = sorted_mj_array[l:]
            pos_window = sorted_mj_array[:len(neg_window)-window_size]
            window = np.concatenate((neg_window, pos_window))
        else:
            window = sorted_mj_array[l:r]

        # point is a 1 x 4 array, with x,y,z and theta from origin,
        # now adding the displacement, so now each item is 1x2,
        # with the first element consisting of previous 1x4 array
        window_points_by_displacement = [
            np.array(point, np.linalg.norm(point - centroid)) for point in window]

        min_index = np.argmin(window_points_by_displacement[:, 1])
        min_element = window_points_by_displacement.pop(min_index)[0][:4]
        nearest.append(window_points_by_displacement.pop(min_index))

        max_index = np.argmax(window_points_by_displacement[:, 1])
        max_element, max_elem_displacement = window_points_by_displacement.pop(
            max_index)
        max_point = max_element[:3]
        furthest.append([max_point, max_elem_displacement])

        # window_points going into median now only consists of what remains after popping min and max
        median.extend(window_points_by_displacement)

        if r > len(sorted_mj_array) + end_r:
            break
        r += 1
        l += 1
    return (np.unique(nearest), np.unique(median), np.unique(furthest))


def avg_displacement(pts_with_displacement):
    return np.sum(pts_with_displacement[:, 1])//len(pts_with_displacement)


def angle_between(v1, v2):
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
    pprint(mj_path_array)

    smoothed_mvj_points = v2m.smooth_landmarks(mj_path_array)
    pprint(smoothed_mvj_points)

    sorted_smoothed_mvj_points, centroid = sort_points_by_angle(
        smoothed_mvj_points)
    nearest, median, furthest = partition_by_displacement(
        sorted_smoothed_mvj_points, centroid, 3)

    print("avg displacement of closest points", avg_displacement(nearest))
    print("avg displacement of median points", avg_displacement(median))
    print("avg displacement of furthest points", avg_displacement(furthest))
