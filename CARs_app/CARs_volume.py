import cv2
import poseDetectionModule as pm
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import open3d as o3d
from pprint import pprint
import time
import json
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

    return longest_radius, idealized_joint_center, final_mj_path


def partition_mj_path(jt_center, mj_path):
    epsilon = .01
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
        q_key += "1" if pt[0] > (jt_center[0]+epsilon) else "0"
        q_key += "1" if pt[1] > (jt_center[1]+epsilon) else "0"
        q_key += "1" if pt[2] > (jt_center[2]+epsilon) else "0"
        by_quadrant[k_to_q[q_key]].append(pt)
    return by_quadrant

# this doesn't work that well....


def denoise_points(mj_path):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(mj_path)

    # returns the points AND the index of each (remaining) point
    filtered_cloud, _ = point_cloud.remove_statistical_outlier(
        nb_neighbors=5, std_ratio=3.0)

    denoised_points = np.asarray(filtered_cloud.points)

    print(f"Denoisifying removed {len(mj_path)-len(denoised_points)}")

    return denoised_points

# now trying to just do a smooth curve fit...


def smooth_path(mj_path, poly_degree):
    x = mj_path[:, 0]
    y = mj_path[:, 1]
    z = mj_path[:, 2]
    coefficients_y = np.polynomial.polynomial.polyfit(
        x, y, poly_degree)
    coefficients_z = np.polynomial.polynomial.polyfit(
        x, z, poly_degree)
    fitted_y = np.polynomial.polynomial.polyval(x, coefficients_y)
    fitted_z = np.polynomial.polynomial.polyval(x, coefficients_z)

    return (x, fitted_y, fitted_z)


def plot_CARs(avg_radius, jt_center, mvj_path_by_quadrant, sphere=True, show_smoothed=False):
    # WILL NEED TO HAND IN MORE ARRAYs if want to see other points
    # xc, yc, zc = tj_array[:, 0], tj_array[:, 1], tj_array[:, 2]
    # xo, yo, zo = mj_array[:, 0], mj_array[:, 1], mj_array[:, 2]
    color_list = 'bgrcmykw'
    fig = plt.figure(figsize=(10, 10), facecolor="w")
    ax = plt.axes(projection="3d", aspect="equal")
    all_mv_points = []
    for q in range(8):
        mvj_path = np.array(mvj_path_by_quadrant[q])
        if len(mvj_path) == 0:
            continue

        xn, yn, zn = mvj_path[:, 0], mvj_path[:, 1], mvj_path[:, 2]

        # scatter_plot = ax.scatter(xo, yo, zo, marker='^')
        ax.scatter(xn, yn, zn, s=10, marker='o', color=color_list[q])
        # scatter_plot = ax.scatter(xc, yc, zc, marker='x')
        if show_smoothed:
            all_mv_points.extend(mvj_path)

    if sphere:
        phi = np.linspace(0, np.pi, 20)
        theta = np.linspace(0, 2 * np.pi, 40)
        x = avg_radius * np.outer(np.sin(theta), np.cos(phi)) + jt_center[0]
        y = avg_radius * np.outer(np.sin(theta), np.sin(phi)) + jt_center[1]
        z = avg_radius * np.outer(np.cos(theta),
                                  np.ones_like(phi)) + jt_center[2]

        ax.plot_wireframe(x, y, z, color='k', linewidth=0.3,
                          rstride=1, cstride=1)

    if show_smoothed:
        all_pts = np.array(all_mv_points)
        fitted_x, fitted_y, fitted_z = smooth_path(all_pts, 15)
        plt.plot(fitted_x, fitted_y, fitted_z, 'r-', label='fitted curve')

    ax.scatter(0, 0, 0, marker='X', s=50, color='green')
    ax.scatter(
        jt_center[0], jt_center[1], jt_center[2], s=50, color='#f07b2e')
    plt.title(
        f"Display of Moving Joint Path around Target Joint", fontsize=30)
    ax.set_xlabel('X_values', fontweight='bold')
    ax.set_ylabel('Y_values', fontweight='bold')
    ax.set_zlabel('Z_values', fontweight='bold')
    ax.set_xlim3d(-0.5, 0.5)
    ax.set_ylim3d(0, -1.0)
    ax.set_zlim3d(-0.5, 0.5)

    # outfile = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/full_CAR_lower_threshold.png"
    outfile = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/full_CAR_higher_threshold.png"
    # plt.savefig(outfile)
    plt.show()


def calc_CARs_volume(jt_center, mj_path_array):
    q1_pts = []
    q2_pts = []
    q3_pts = []
    q4_pts = []
    q5_pts = []
    q6_pts = []
    q7_pts = []
    q8_pts = []
    for pt in mj_path_array:
        # determining q1
        if pt[0] > jt_center[0] and pt[1] > jt_center[1] and pt[2] > jt_center[2]:
            q1_pts.append(pt)
        # determining q2
        elif pt[0] < jt_center[0] and pt[1] > jt_center[1] and pt[2] > jt_center[2]:
            q2_pts.append(pt)
        # determining q3
        elif pt[0] < jt_center[0] and pt[1] > jt_center[1] and pt[2] < jt_center[2]:
            q3_pts.append(pt)
        # determining q4
        elif pt[0] > jt_center[0] and pt[1] > jt_center[1] and pt[2] < jt_center[2]:
            q4_pts.append(pt)
        # determining q5
        elif pt[0] > jt_center[0] and pt[1] < jt_center[1] and pt[2] > jt_center[2]:
            q5_pts.append(pt)
        # determining q6
        elif pt[0] < jt_center[0] and pt[1] < jt_center[1] and pt[2] > jt_center[2]:
            q6_pts.append(pt)
        # determining q7
        elif pt[0] < jt_center[0] and pt[1] < jt_center[1] and pt[2] < jt_center[2]:
            q7_pts.append(pt)
        # determining q8
        elif pt[0] > jt_center[0] and pt[1] < jt_center[1] and pt[2] < jt_center[2]:
            q8_pts.append(pt)
    print(q1_pts)


def compute_CARs_hull(target_joint_pose_id, moving_joint_pose_id, real_landmarks, draw=True):
    target_workspace_lms = []
    for frame in real_landmarks:
        # NEED TO handle if point is not in frame (just skip)
        target_joint = frame[target_joint_pose_id][1:]
        target_workspace_lms.append(target_joint)
        moving_bone_end = frame[moving_joint_pose_id][1:]
        target_workspace_lms.append(moving_bone_end)
    points = np.array(target_workspace_lms, dtype=float)
    workspace = ConvexHull(points)
    volume = workspace.volume

    if draw == True:
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        # plt.style.use('seaborn')
        fig = plt.figure(figsize=(20, 10), facecolor="w")
        ax = plt.axes(projection="3d")
        scatter_plot = ax.scatter3D(x, y, z)
        for simplex in workspace.simplices:
            ax.plot3D(points[simplex, 0], points[simplex, 1],
                      points[simplex, 2], 's-')

        plt.title(f"Workspace of joint {target_joint_pose_id}", fontsize=30)
        ax.set_xlabel('X_values', fontweight='bold')
        ax.set_ylabel('Y_values', fontweight='bold')

        plt.show()

    return volume


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
    denoised_pts = denoise_points(mj_path_array)
    by_quadrant = partition_mj_path(jt_center, denoised_pts)
    plot_CARs(avg_radius, jt_center, by_quadrant,
              sphere=True, show_smoothed=True)
    # calc_CARs_volume(jt_center, mj_path_array)

    # result_volume = compute_CARs_volume(12, 14, real_lm_list)
    # print(result_volume)
