import cv2
import time
from pprint import pprint
import poseModule as pm

cap = cv2.VideoCapture(
    "/Users/williamhbelew/Hacking/ocv_playground/pose_est_tutorial/poseEstVids/artem-30fps.mp4")
real_lmList = []
pTime = 0
detector = pm.poseDetector()
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img, draw=True)
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
pprint(real_lmList)
