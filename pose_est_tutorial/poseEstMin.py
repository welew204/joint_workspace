import cv2
# video manipulation
import mediapipe as mp
# allows for pose estimation
import time
from pprint import pprint

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()
# learn more about default values here (relating to finding the joint to track, then the confidence of tracking that point)
# here: https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md

cap = cv2.VideoCapture(
    "/Users/williamhbelew/Hacking/ocv_playground/pose_est_tutorial/poseEstVids/allan-25fps.mp4")

pTime = 0
landmarks_to_print = []
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    landmarks_to_print.append(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks,
                              mpPose.POSE_CONNECTIONS)

    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        # lm is float repr 'ratio of image'
        # print(id, lm)
        # to get pixel value...
        cx, cy = int(lm.x*w), int(lm.y*h)
        cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

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
# pprint(landmarks_to_print)
