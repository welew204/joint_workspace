import CARs_volume as cars
import numpy as np

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

# ----------


def calc_reachable_area(sorted_mj_points, centroid):
    total_reach = 0
    for i,  pt in enumerate(sorted_mj_points):
        reach = np.linalg.norm(pt - centroid)
        # next, want to calc the area of a triangle (speherical triangle?) if I use neighboring points
        # also, re-form partition function as a sorting function that id's what zone a point is in, to tally avgs (and eventually SAs) by per zone
        total_reach += reach
    return total_reach // len(sorted_mj_points)


def sort_points_by_angle(mj_cart_pts):
    """Input smoothed array of points; 
    Outputs sorted array of moving joint points in order of rotation from intial point"""
    centroid = cars.find_centroid(mj_cart_pts)
    origin_rotational_angle = np.empty()
    output_array = np.empty()
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
    r = l + (window_size-1)
    end_r = r
    while l < len(sorted_mj_array - (window_size // 2)):
        if l < 0:
            neg_window = sorted_mj_array[l:]
            pos_window = sorted_mj_array[:r]
            window = neg_window + pos_window
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

        if r == len(sorted_mj_array):
            r = -1
        elif r > len(sorted_mj_array) + end_r:
            break
        r += 1
        l += 1
    return [np.unique(furthest), np.unique(nearest), np.unique(median)]


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
