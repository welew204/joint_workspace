import cv2
import poseDetectionModule as pm
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import open3d as o3d
from pprint import pprint
import time
import json
import math
from collections import defaultdict

# handle video ingest


def new_CARs_vid(camera_int, filename):
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
    epsilon = .001
    k_to_q = {"000": 0,  # hi-flex-abd
              "001": 1,  # low-flex-abd
              "010": 2,  # low-ext-abd
              "011": 3,  # low-ext-add
              "100": 4,  # low-flex-add
              "101": 5,  # hi-flex-add
              "110": 6,  # hi-ext-add
              "111": 7}  # hi-ext-abd
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


def scale_points_to_surface(point_path, avg_radius):
    """point_path is a np.array of 3d points *normalized* around origin!"""
    current_magnitudes = np.linalg.norm(point_path, axis=1)
    scaling_factors = avg_radius / current_magnitudes
    scaled_points = (point_path *
                     scaling_factors[:, np.newaxis])
    return scaled_points


def plot_sphere(ax, rad_avg):
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

    plt.show()


def find_centroid(quadrant_mvj_path):
    """Takes an np.array() as input, returns a single 3d vector as np.array()"""
    length, dim = np.shape(quadrant_mvj_path)

    avg_x = np.sum(quadrant_mvj_path[:, 0])/length
    avg_y = np.sum(quadrant_mvj_path[:, 1])/length
    avg_z = np.sum(quadrant_mvj_path[:, 2])/length
    centroid = np.array([avg_x, avg_y, avg_z])
    return centroid


def calc_spherical_surface_area(coords, radius):
    # given coords in (lat, long) in radians
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
        area += (x2 - x1) * (2 + np.sin(y1) + np.sin(y2))
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
    lat = np.arcsin(coords_cart[2] / radius_of_sphere)
    lon = np.arctan2(coords_cart[1], coords_cart[0])
    return lat, lon


def draw_all_points_on_sphere(avg_radius, jt_center, all_mvj_points):
    # original path points of TOTAL joint path motion
    ax = plt.axes(projection="3d", aspect="equal")
    ax.set_xlim3d(-.75, .75)
    ax.set_ylim3d(0, -1.5)
    ax.set_zlim3d(-.75, .75)
    ax.set_xlabel('X_values', fontweight='bold')
    ax.set_ylabel('Y_values', fontweight='bold')
    ax.set_zlabel('Z_values', fontweight='bold')

    mjp_x, mjp_y, mjp_z = all_mvj_points[:,
                                         0], all_mvj_points[:, 1], all_mvj_points[:, 2]
    # ax.scatter(mjp_x, mjp_y, mjp_z, s=2, marker='.', color="black")
    ax.scatter(jt_center[0], jt_center[1], jt_center[2],
               s=20, marker='X', color="red")
    normalized_all_pts = all_mvj_points - jt_center
    # these scaled_points are oriented around Origin (0,0,0)
    scaled_points = scale_points_to_surface(normalized_all_pts,
                                            avg_radius)
    # converting points to lat-lon
    all_pts_as_lat_lon = [convert_cartesian_to_latlon_rad(
        pt, avg_radius) for pt in scaled_points]
    full_path_SA = calc_spherical_surface_area(all_pts_as_lat_lon, avg_radius)
    print("Surface area of visited region: ", abs(full_path_SA))

    # moving scale dpoints back to be around jt_center...
    scaled_points_to_plot = scaled_points + jt_center
    # plotting them
    sp_x, sp_y, sp_z = scaled_points_to_plot[:,
                                             0], scaled_points_to_plot[:, 1], scaled_points_to_plot[:, 2]
    ax.scatter(sp_x, sp_y, sp_z, s=15, marker='o',
               color='orange')
    plot_sphere(ax, avg_radius)
    plt.show()


# map resultant workspace-zone onto video??
if __name__ == "__main__":
    # new_CARs_vid(1, 'R_GH_partial_improved_tracking')
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
