import cv2
import poseDetectionModule as pm
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from pprint import pprint
import time

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
    while True:
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


def compute_CARs_volume(target_joint_pose_id, moving_joint_pose_id, real_landmarks, draw=True):
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
    # new_CARs_vid(1, 'R_gh')
    vid_path = "/Users/williamhbelew/Hacking/ocv_playground/CARs_app/sample_CARs/R_gh_output.mp4"
    real_lm_list = process_CARs_vid(vid_path)
    result_volume = compute_CARs_volume(12, 14, real_lm_list)
    print(result_volume)
