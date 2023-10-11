import cv2
import poseDetectionModule as pm
import numpy as np

import matplotlib.pyplot as plt

from pprint import pprint
import time
import json
import math
from collections import defaultdict

# handle video ingest


def new_CARs_vid(camera_int, filename):
    """builds new file from device camera stream (given by 'camera_int'), 
    saves in given filepath as mp4v"""
    cap = cv2.VideoCapture(camera_int)

    # size must be defined according to capture (?)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    # define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    # fourcc = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
    out = cv2.VideoWriter(
        f'CARs_app/sample_CARs/{filename}_output.mp4', fourcc, 20.0, size)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't recieve frame (stream end?). Exiting...")
            break

        # write the frame
        out.write(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(10) == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def process_CARs_vid(vid_path):
    """OLD flow for generating cv2 frames from video (from given vid_path) // output: real_landmark_list"""
    cap = cv2.VideoCapture(vid_path)
    real_lmList = []
    pTime = 0
    detector = pm.poseDetector()
    start_time = time.time()
    while True:
        # this just allows for a time-capped run of the processing...WHAT'S A BETTER WAY TO HANDLE THIS?
        # if pTime - start_time > 15.0:
        #     break
        success, img = cap.read()
        img = detector.findPose(img)
        # lmList = detector.findPosition(img, draw=True)
        intermediateRL_landmarks = detector.findRealPosition(img)
        real_lmList.append(intermediateRL_landmarks)
        # pprint(lmList[14])
        # draw a specific joint ONLY (set draw to False, above)
        # cv2.circle(img, (lmList[14][1], lmList[14][2]),
        # 25, (0, 0, 255), cv2.FILLED)
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Image", img)

        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return real_lmList
# select joint specific points
# compute np.convex_hull.volume


def normalize_joint_center(real_landmarks, target_joint_id, moving_joint_id):
    tj_path = []
    mj_path = []
    for frame, lm_list in enumerate(real_landmarks):
        tj_vals = lm_list[target_joint_id][1:]
        mj_vals = lm_list[moving_joint_id][1:]
        tj_path.append(tj_vals)
        mj_path.append(mj_vals)
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


def partition_mj_path(jt_center, mj_path):
    epsilon = .001  # make larger if I want to treat close values as boundary values
    # trying to correlate these w. color! aka ... color_list = 'bgrcmykw'
    k_to_q = {"000": 0,
              "001": 1,
              "010": 2,
              "011": 3,
              "100": 4,
              "101": 5,
              "110": 6,
              "111": 7}
    by_quadrant = defaultdict(list)
    for i, pt in enumerate(mj_path):
        q_key = ""
        # old version that was skewing...
        q_key += "1" if pt[0] > (jt_center[0]+epsilon) else "0"
        q_key += "1" if pt[1] > (jt_center[1]+epsilon) else "0"
        q_key += "1" if pt[2] > (jt_center[2]+epsilon) else "0"
        # new version that is miscategorizing
        # q_key += "1" if np.abs(pt[0]-jt_center[0]) > epsilon else "0"
        # q_key += "1" if np.abs(pt[1]-jt_center[1]) > epsilon else "0"
        # q_key += "1" if np.abs(pt[2]-jt_center[2]) > epsilon else "0"
        by_quadrant[k_to_q[q_key]].append(pt)
    return by_quadrant

# these 4 following functions are crucial for a per-zone analysis of the volume


def in_order_partition_mj_path(jt_center, mj_path):
    epsilon = .001
    k_to_q = {
        "100": 1,  # purple (m)
        "101": 2,  # yellow (y)
        "001": 3,  # black (k)
        "011": 4,  # green (g)
        "111": 5,  # red (r)
        "010": 6,  # blue (b)
        "110": 7,  # cyan (c)
        "000": 8   # white (w)
    }
    in_path_order = {
        1: [],  # purple (m)
        2: [],  # yellow (y)
        3: [],  # black (k)
        4: [],  # green (g)
        5: [],  # red (r)
        6: [],  # blue (b)
        7: [],  # cyan (c)   # white (w)
    }
    for i, pt in enumerate(mj_path):
        # FIRST filter out points too close to tell
        # THEN tag remaining points into correct zone
        q_key = ""
        q_key += "1" if pt[0] > (jt_center[0] + epsilon) else "0"
        q_key += "1" if pt[1] > (jt_center[1] + epsilon) else "0"
        q_key += "1" if pt[2] > (jt_center[2] + epsilon) else "0"
        # should I be adding the epsilon to the POINT?
        in_path_order[k_to_q[q_key]].append(pt)
    return in_path_order


def detect_plane_crossed(pt1, pt2):
    """given 2 points on either side of an origin plane, determine which plane they cross;
    return the normal of that plane"""
    inter = pt2 - pt1
    if min(pt1[0], pt2[0]) < 0 < max(pt1[0], pt2[0]):
        # crossed the z-y plane, where x = 0
        plane_normal = np.array(1, 0, 0)
    elif min(pt1[1], pt2[1]) < 0 < max(pt1[1], pt2[1]):
        # crossed the x-z plane, where y = 0
        plane_normal = np.array(0, 1, 0)
    if min(pt1[2], pt2[2]) < 0 < max(pt1[2], pt2[2]):
        # crossed the x-y plane, where z = 0
        plane_normal = np.array(0, 0, 1)
    return plane_normal


def get_intersection_w_plane(plane_normal, intervening_vector, jt_center):
    epsilon = 1e-6
    inter_norm = np.linalg.norm(intervening_vector)
    unit_inter = intervening_vector/inter_norm
    inter_point = intervening_vector
    ndotu = plane_normal.dot(unit_inter)
    if abs(ndotu) < epsilon:
        raise ValueError(
            "interpolating vector is parallel to or on intervening plane")
    # jt_center is just a stand-in for 'any point on plane' since it will be on all intervening planes
    w = inter_point - jt_center
    si = -plane_normal.dot(w) / ndotu
    Psi = w + si * unit_inter + jt_center

    return Psi


def add_interpolated_and_pivots(ordered_mj_path, jt_center, avg_radius):
    pivots = {
        # the first point is only needed if zone 2 is visited (otherwise, only second pivot is needd)
        1: [(0, -1, 0), (0, 0, -1)],
        2: [(0, -1, 0)],
        3: [(-1, 0, 0), (0, -1, 0)],
        # second point only needed if zone 5 is visited!
        4: [(-1, 0, 0), (0, 1, 0)],
        5: [(0, 1, 0)],
        6: [(0, 0, -1), (-1, 0, 0)],  # even single point entries here are ok
        # because the interpolation will add 2 more 'boundary' points,
        # which when combined w/ these pivots will make for a nice polygon
        7: [(1, 0, 0), (0, 0, -1)]
    }
    # sort the partitioned mvj_path dict in zone order:
    # TODO redo with python.collections.ordered_dict
    zone_keys = list(ordered_mj_path.keys()).sort()
    sorted_ord_mj_path = {i: ordered_mj_path[i] for i in zone_keys}
    prev_key = zone_keys[-1]
    for z, pts in sorted_ord_mj_path.items():
        # get first interpolated point
        principle = pts[0]
        prev_point = sorted_ord_mj_path[prev_key][-1]

# RIKS IDEA -----------
# convert each point to lat/lon
# get the average of the lats, lons to find 'linear interpolation'
# convert THIS point BACK to (x,y,z)

        inter_vector1 = principle - prev_point
        # this subtraction is intended to normalize around the origin to determine plane crossed
        plane_normal = detect_plane_crossed(
            principle-jt_center, prev_point-jt_center)
        inter_point_1 = get_intersection_w_plane(
            plane_normal, inter_vector1, jt_center)
        # add to FRONT of zone/octant z array

        sorted_ord_mj_path[z].insert(0, inter_point_1)
        # get second interpolated point >> this will be handled as the function iterates

        sorted_ord_mj_path[prev_key].append(inter_point_1)
        # add pivots (checking for next zone existence as needed), for both current z zone and prev_key zone; add to END so that the FIRST is still valid
        zone_pivots = pivots[z]
        for piv in zone_pivots:
            # check for len == 0 for next key in dict, IF TRUE only needs sec, else, needs both
            sorted_ord_mj_path[z].append(np.array(piv)*avg_radius + jt_center)
        # update prev_key value


def scale_points_to_surface(point_path, avg_radius):
    """point_path is a np.array of 3d points *normalized* around origin, 
    tho not spherical; returns points scaled to the surface of a sphere"""
    current_magnitudes = np.linalg.norm(point_path, axis=1)
    scaling_factors = avg_radius / current_magnitudes
    scaled_points = (point_path *
                     scaling_factors[:, np.newaxis])
    return scaled_points


def plot_sphere(ax, rad_avg, jt_center):
    """plots a wire-mesh sphere on the axis of size given by rad_avg"""
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = rad_avg * np.outer(np.sin(theta), np.cos(phi)) + jt_center[0]
    y = rad_avg * np.outer(np.sin(theta), np.sin(phi)) + jt_center[1]
    z = rad_avg * np.outer(np.cos(theta),
                           np.ones_like(phi)) + jt_center[2]

    ax.plot_wireframe(x, y, z, color='k', linewidth=0.1,
                      rstride=1, cstride=1)

    # adding some axial planes to see better

    # Create a meshgrid for the x, y, and z coordinates of the planes
    p1x, p1y = np.meshgrid(np.linspace(jt_center[0]-rad_avg, jt_center[0]+rad_avg, 10),
                           np.linspace(jt_center[1]-rad_avg, jt_center[1]+rad_avg, 10))
    p1z = jt_center[2] * np.ones_like(p1x)
    p2x, p2z = np.meshgrid(np.linspace(jt_center[0]-rad_avg, jt_center[0]+rad_avg, 10),
                           np.linspace(jt_center[2]-rad_avg, jt_center[2]+rad_avg, 10))
    p2y = jt_center[1] * np.ones_like(p2x)
    p3y, p3z = np.meshgrid(np.linspace(jt_center[1]-rad_avg, jt_center[1]+rad_avg, 10),
                           np.linspace(jt_center[2]-rad_avg, jt_center[2]+rad_avg, 10))
    p3x = jt_center[0] * np.ones_like(p3y)

    # Create the figure and axes

    # Plot the planes
    ax.plot_surface(p1x, p1y, p1z, alpha=0.2, color='blue')
    ax.plot_surface(p2x, p2y, p2z, alpha=0.2, color='red')
    ax.plot_surface(p3x, p3y, p3z, alpha=0.2, color='green')


def add_nose_and_other_gh(ax):
    """plots a nose (joint 0) and opposing shoulder (joint 11) on the given axis;"""
    nose_coord = []
    for frame in sample1:
        nose = frame[0]
        hx, hy, hz = nose[1:]
        nose_coord.append([hx, hy, hz])
    nose_coord = np.array(nose_coord)
    nose_coord_avg = find_centroid(nose_coord)
    ax.scatter(nose_coord_avg[0], nose_coord_avg[1],
               nose_coord_avg[2], marker='X', s=50, color='yellow')
    # left shoulder
    lGH_coord = []
    for frame in sample1:
        lGH = frame[11]
        hx, hy, hz = lGH[1:]
        lGH_coord.append([hx, hy, hz])
    lGH_coord = np.array(lGH_coord)
    lGH_coord_avg = find_centroid(lGH_coord)
    ax.scatter(lGH_coord_avg[0], lGH_coord_avg[1],
               lGH_coord_avg[2], marker='X', s=50, color='orange')


def improved_add_nose_and_other_gh(ax, all_lms):
    """plots a nose (joint 0) and opposing shoulder (joint 11) on the given axis;"""
    nose_coord = []
    for frame in all_lms:
        nose = frame[0][0]
        hx, hy, hz = nose["x"], nose["y"], nose["z"]
        nose_coord.append([hx, hy, hz])
    nose_coord = np.array(nose_coord)
    nose_coord_avg = find_centroid(nose_coord)
    ax.scatter(nose_coord_avg[0], nose_coord_avg[1],
               nose_coord_avg[2], marker='X', s=50, color='yellow')
    # left shoulder
    lGH_coord = []
    for frame in all_lms:
        lGH = frame[0][11]
        hx, hy, hz = lGH["x"], lGH["y"], lGH["z"]
        lGH_coord.append([hx, hy, hz])
    lGH_coord = np.array(lGH_coord)
    lGH_coord_avg = find_centroid(lGH_coord)
    ax.scatter(lGH_coord_avg[0], lGH_coord_avg[1],
               lGH_coord_avg[2], marker='X', s=50, color='green')


def improved_add_pelvis_and_other_hip(ax, all_lms):
    """plots a pelvis (at real-origin) and opposing hip (joint 23) on the given axis;"""
    # pelvis (origin)
    ax.scatter(0, 0, 0, marker='X', s=50, color='yellow')
    # left hip
    l_hip_coord = []
    for frame in all_lms:
        l_hip = frame[0][23]
        hx, hy, hz = l_hip["x"], l_hip["y"], l_hip["z"]
        l_hip_coord.append([hx, hy, hz])
    l_hip_coord = np.array(l_hip_coord)
    l_hip_coord_avg = find_centroid(l_hip_coord)
    ax.scatter(l_hip_coord_avg[0], l_hip_coord_avg[1],
               l_hip_coord_avg[2], marker='X', s=50, color='green')


def plot_on_sphere(subplot_id, jt_center, mvj_path_quadrant, sphere=True):
    color_list = 'bgrcmykw'
    ax = plt.axes(projection="3d", aspect="equal")
    ax.set_xlim3d(-.75, .75)
    ax.set_ylim3d(0, -1.5)
    ax.set_zlim3d(-.75, .75)
    ax.set_xlabel('X_values', fontweight='bold')
    ax.set_ylabel('Y_values', fontweight='bold')
    ax.set_zlabel('Z_values', fontweight='bold')

    mvj_path = np.array(mvj_path_quadrant)

    # compute avg
    centroid = find_centroid(mvj_path)
    avg_displacement = centroid - jt_center
    rad_avg = np.linalg.norm(avg_displacement)

    xn, yn, zn = mvj_path[:, 0], mvj_path[:, 1], mvj_path[:, 2]
    if sphere:
        phi = np.linspace(0, np.pi, 20)
        theta = np.linspace(0, 2 * np.pi, 40)
        x = rad_avg * np.outer(np.sin(theta), np.cos(phi)) + jt_center[0]
        y = rad_avg * np.outer(np.sin(theta), np.sin(phi)) + jt_center[1]
        z = rad_avg * np.outer(np.cos(theta),
                               np.ones_like(phi)) + jt_center[2]

        ax.plot_wireframe(x, y, z, color='k', linewidth=0.3,
                          rstride=1, cstride=1)

        # adding some axial planes to see better

        # Create a meshgrid for the x, y, and z coordinates of the planes
        p1x, p1y = np.meshgrid(np.linspace(jt_center[0]-rad_avg, jt_center[0]+rad_avg, 10),
                               np.linspace(jt_center[1]-rad_avg, jt_center[1]+rad_avg, 10))
        p1z = jt_center[2] * np.ones_like(p1x)
        p2x, p2z = np.meshgrid(np.linspace(jt_center[0]-rad_avg, jt_center[0]+rad_avg, 10),
                               np.linspace(jt_center[2]-rad_avg, jt_center[2]+rad_avg, 10))
        p2y = jt_center[1] * np.ones_like(p2x)
        p3y, p3z = np.meshgrid(np.linspace(jt_center[1]-rad_avg, jt_center[1]+rad_avg, 10),
                               np.linspace(jt_center[2]-rad_avg, jt_center[2]+rad_avg, 10))
        p3x = jt_center[0] * np.ones_like(p3y)

        # Create the figure and axes

        # Plot the planes
        ax.plot_surface(p1x, p1y, p1z, alpha=0.2, color='blue')
        ax.plot_surface(p2x, p2y, p2z, alpha=0.2, color='red')
        ax.plot_surface(p3x, p3y, p3z, alpha=0.2, color='green')

    # plot the radius (to the centroid)...
    ax.plot([centroid[0], jt_center[0]], [
        centroid[1], jt_center[1]], zs=[centroid[2], jt_center[2]])

    # calculate + scale each vector to the surface of the sphere...
    mvj_normalized_to_joint_center = mvj_path-jt_center
    current_magnitudes = np.linalg.norm(mvj_normalized_to_joint_center, axis=1)
    scaling_factors = rad_avg / current_magnitudes
    scaled_points = (mvj_normalized_to_joint_center *
                     scaling_factors[:, np.newaxis]) + jt_center
    scaled_points_CHECK = scale_points_to_surface(
        mvj_normalized_to_joint_center, rad_avg) + jt_center

    sp_x, sp_y, sp_z = scaled_points[:,
                                     0], scaled_points[:, 1], scaled_points[:, 2]
    # original path points of THIS quadrant
    # ax.scatter(xn, yn, zn, s=5, marker='.', color="black")
    ax.scatter(sp_x, sp_y, sp_z, s=15, marker='o',
               color=color_list[subplot_id-1])

    # plotting some raw_data:
    # origin (aka the middle of the pelvis)
    ax.scatter(0, 0, 0, marker='X', s=50, color='green')
    # original path points of TOTAL joint path motion
    mjp_x, mjp_y, mjp_z = mj_path_array[:,
                                        0], mj_path_array[:, 1], mj_path_array[:, 2]
    ax.scatter(mjp_x, mjp_y, mjp_z, s=5, marker='.', color="black")
    # nose
    add_nose_and_other_gh(ax)

    plt.show()


def find_centroid(quadrant_mvj_path):
    """Takes an np.array() as input, returns a single 3d vector as np.array()"""
    length, dim = np.shape(quadrant_mvj_path)

    avg_x = np.sum(quadrant_mvj_path[:, 0])/length
    avg_y = np.sum(quadrant_mvj_path[:, 1])/length
    avg_z = np.sum(quadrant_mvj_path[:, 2])/length
    centroid = np.array([avg_x, avg_y, avg_z])
    return centroid


def toRadians(angleInDegrees):
    return (angleInDegrees*np.pi) / 180


def calc_spherical_surface_area(coords, radius):
    # given coords in (lat, long) THAT ARE ALREADY IN RADIANS (from convert_cartesian_to_latlon_rad)
    # # !!! order of points matter! they must be in a 'ring'; Clockwise will be positive,
    # CC will be negative
    #  and radius of sphere that shape is on surface of
    # return SA of spherical shape
    area = 0
    num_of_pts = len(coords)
    x1 = coords[num_of_pts-1][0]
    y1 = coords[num_of_pts-1][1]
    for i in range(num_of_pts):
        x2 = coords[i][0]
        y2 = coords[i][1]
        area += (x2 - x1) * \
            (2 + np.sin(y1) + np.sin(y2))
        x1 = x2
        y1 = y2
    area_of_spherical_polygon = (area * radius**2) / 2
    return area_of_spherical_polygon


def convert_latlon_to_cartesian(latlon_rad, radius_of_sphere):
    lat, lon = latlon_rad[0], latlon_rad[1]
    # using this formula: https://stackoverflow.com/a/1185413/19589299
    x = radius_of_sphere * np.cos(lat) * np.cos(lon)
    y = radius_of_sphere * np.cos(lat) * np.sin(lon)
    z = radius_of_sphere * np.sin(lat)
    return np.array([x, y, z])


def convert_cartesian_to_latlon_rad(coords_cart, radius_of_sphere):
    # from this article: https://rbrundritt.wordpress.com/2008/10/14/conversion-between-spherical-and-cartesian-coordinates-systems/
    lat = np.arcsin(coords_cart[2] / radius_of_sphere)
    lon = np.arctan2(coords_cart[1], coords_cart[0])
    return lat, lon


def draw_all_points_on_sphere(avg_radius, jt_center, all_mvj_points, lms_array, scale_to_sphere=False):
    # original path points of TOTAL joint path motion
    ax = plt.axes(projection="3d", aspect="equal")
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(0, -2)
    ax.set_zlim3d(-1, 1)
    ax.set_xlabel('X_values', fontweight='bold')
    ax.set_ylabel('Y_values', fontweight='bold')
    ax.set_zlabel('Z_values', fontweight='bold')

    mjp_x, mjp_y, mjp_z = all_mvj_points[:,
                                         0], all_mvj_points[:, 1], all_mvj_points[:, 2]
    ax.scatter(jt_center[0], jt_center[1], jt_center[2],
               s=20, marker='X', color="red")
    if scale_to_sphere:
        normalized_all_pts = all_mvj_points - jt_center
        # these scaled_points are oriented around Origin (0,0,0)
        scaled_points = scale_points_to_surface(normalized_all_pts,
                                                avg_radius)
        # converting points to lat-lon
        all_pts_as_lat_lon = [convert_cartesian_to_latlon_rad(
            pt, avg_radius) for pt in scaled_points]
        full_path_SA = calc_spherical_surface_area(
            all_pts_as_lat_lon, avg_radius)
        joint_sphere_area = 4*math.pi*(avg_radius**2)
        print("Surface area of visited region: ", abs(full_path_SA))
        print("Surface area of joint sphere: ", joint_sphere_area)
        print("Proportion of visited area: ",
              abs(full_path_SA)/joint_sphere_area)

        # moving scale dpoints back to be around jt_center...
        scaled_points_to_plot = scaled_points + jt_center
        # plotting them
        sp_x, sp_y, sp_z = scaled_points_to_plot[:,
                                                 0], scaled_points_to_plot[:, 1], scaled_points_to_plot[:, 2]
        ax.scatter(sp_x, sp_y, sp_z, s=15, marker='o',
                   color='blue')
    else:
        ax.scatter(mjp_x, mjp_y, mjp_z, s=2, marker='.', color="black")
    plot_sphere(ax, avg_radius, jt_center)
    # turn ON to show GH
    improved_add_nose_and_other_gh(ax, all_lms=lms_array)
    # turn ON to show hip
    # improved_add_pelvis_and_other_hip(ax, all_lms=lms_array)
    plt.show()


def plot_all_partitioned_on_sphere(mvj_path, jt_center, avg_radius):
    plt.clf()
    color_list = 'bgrcmykw'
    ax = plt.axes(projection="3d", aspect="equal")
    ax.set_xlim3d(-.75, .75)
    ax.set_ylim3d(0, -1.5)
    ax.set_zlim3d(-.75, .75)
    ax.set_xlabel('X_values', fontweight='bold')
    ax.set_ylabel('Y_values', fontweight='bold')
    ax.set_zlabel('Z_values', fontweight='bold')
    # start w/ all points
    by_quadrant = partition_mj_path(jt_center, mj_path_array)
    i = 0
    for q, pts in by_quadrant.items():
        normalized_pts = pts - jt_center
        scaled_points = scale_points_to_surface(normalized_pts,
                                                avg_radius) + jt_center
        px, py, pz = scaled_points[:,
                                   0], scaled_points[:, 1], scaled_points[:, 2]
        ax.scatter(px, py, pz, s=10, marker='o', color=color_list[i])
        i += 1
    plot_sphere(ax, avg_radius, jt_center)

    ax.scatter(jt_center[0], jt_center[1], jt_center[2],
               s=20, marker='X', color="red")

    # partion
    # for each partion, plot onto the same sphere
    # add nose/gh reference
    # add joint ctr referecne
    add_nose_and_other_gh(ax)

    plt.show()


def plot_partiton_calc_area(partition_pts, color, plot=True):
    """this takes in a single octant of points (including 4 additional:
    - 2 interpolated_points (one at beginning, one at end of array)
    - 2 pivot_point (added at end, so that the first point follows the final pivot)), 
    a specified color and optional 'plot' argument (def: True);
    returns the surface area of that octants visited surface area"""


# map resultant workspace-zone onto video??
if __name__ == "__main__":
    capture_CAR_from_camera = True
    new_vid_filename = 'R_hip_quadruped_small'
    if capture_CAR_from_camera:
        new_CARs_vid(1, new_vid_filename)
        exit()
    vid_path = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/sample_CARs/R_gh_bare_output.mp4"
    # real_lm_list = process_CARs_vid(vid_path)

    # the below was part of my validation that every run created the same path
    """ right_now = time.time()
    with open('sample_landmarks.json', 'r') as landmark_json:
        vid_runs = json.load(landmark_json)
    vid_runs[right_now] = real_lm_list
    with open('sample_landmarks.json', 'w') as landmark_json:
        json.dump(vid_runs, landmark_json) """

    """ KEY for sample landmarks:
    - 4 = full, bareskin GH CAR
    - 5 = partial, in jacket
    - 6 = partial, in tshirt, w/ higher confidence thresholds (.8)
    - 7 = partial, in tshirt, w/ higher confidence thresholds (0.5, 0.9) >> any higher and no landmarks are detected
    - 8 = full, bareskin GH CAR, w/ higher confidence thresholds (0.5, 0.9)
      """

    with open('sample_landmarks.json', 'r') as landmark_json:
        vid_runs = json.load(landmark_json)
    sample1 = vid_runs[list(vid_runs.keys())[8]]
    avg_radius, jt_center, mj_path_array = normalize_joint_center(
        sample1, 12, 14)
    by_quadrant = partition_mj_path(jt_center, mj_path_array)
    i = 1
    # trying out running the entire path of points...
    # plot_on_sphere(i, jt_center, mj_path_array)
    # ^^ this should really be done in a new function that:
    # > calculates the avg radius differently (right now it's really small ~.1)

    plot_all_partitioned_on_sphere(mj_path_array, jt_center, avg_radius)

    # plotting all points onto the sphere, calculating SA
    draw_all_points_on_sphere(avg_radius, jt_center, mj_path_array)

    for q, pts in by_quadrant.items():
        """if len(pts) < 5:
            i += 1
            continue"""
        # plot_by_quadrant(i, jt_center, pts, planar_tangent=True)
        plot_on_sphere(i, jt_center, pts)
        i += 1
        print("hi Dr.DD")
        plt.clf()
    plt.legend(loc="best")

    # save the plt to a file!

    plt.show()

    # plot_CARs(avg_radius, jt_center, by_quadrant, sphere=True, show_smoothed=False)
    # calc_CARs_volume(jt_center, mj_path_array)

    # result_volume = compute_CARs_volume(12, 14, real_lm_list)
    # print(result_volume)
