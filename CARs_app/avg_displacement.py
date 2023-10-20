import CARs_volume as cars
import numpy as np

# take in an array of normalized and scaled points (same radius)

# calculate centroid of all points, ie: the central axis for all calcs
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


def sort_points_by_angle(mj_cart_pts):
    """Input smoothed array of points; 
    Outputs sorted array of moving joint points in order of rotation from intial point"""
    centroid = cars.find_centroid(mj_cart_pts)
    origin_rotational_angle = np.empty()
    output_array = np.empty()
    for i, pt in enumerate(mj_cart_pts):
        dist_from_centroid = centroid - pt
        if i == 0:
            origin_rotational_angle = dist_from_centroid
            continue
        theta = angle_between(origin_rotational_angle, pt)
        output_array.append(np.array(pt, theta))
    return output_array.sort(key=(lambda x: x[1])), centroid


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
