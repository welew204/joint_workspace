import cv2
# video manipulation
import mediapipe as mp
# allows for pose estimation
import time
from pprint import pprint

# Create class that detects pose, finds neccessary points


class poseDetector():
    def __init__(self, mode=False, modelComplexity=2, smooth_landmarks=True, detectionConfidence=0.5, trackConfidence=0.9):
        self.mode = mode
        self.modelComplexity = modelComplexity
        self.smooth = smooth_landmarks
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     model_complexity=self.modelComplexity,
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionConfidence,
                                     min_tracking_confidence=self.trackConfidence)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # lm is float repr 'ratio of image'
                # print(id, lm)
                # to get pixel value...
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
        return lmList

    def findRealPosition(self, img):
        meters_lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_world_landmarks.landmark):
                meters_lmList.append([id, lm.x, lm.y, lm.z])

        return meters_lmList

# pprint(landmarks_to_print)


def main():
    cap = cv2.VideoCapture(
        "/Users/williamhbelew/Hacking/ocv_playground/pose_est_tutorial/poseEstVids/artem-30fps.mp4")

    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        pprint(lmList[14])
        # draw a specific joint ONLY (set draw to False, above)
        cv2.circle(img, (lmList[14][1], lmList[14][2]),
                   25, (0, 0, 255), cv2.FILLED)
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


if __name__ == "__main__":
    main()
