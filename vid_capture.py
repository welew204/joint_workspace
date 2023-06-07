import numpy as np
import cv2 as cv


def vid_file_processing(filepath):
    cap = cv.VideoCapture(
        filepath)
    # 0 == phone camera (?)
    # 1 == laptop camera
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        # capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't recieve frame (stream end?). Exiting...")
            break

        # our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # display the resulting frame
        cv.imshow('frame', gray)
        if cv.waitKey(10) == ord('q'):
            # number here is in milliseconds; lower will be FAST, higher (>30) will be SLOWMO
            break

    # when everything is done, release the capture
    cap.release()
    cv.destroyAllWindows()


def vid_file_cap_flip_save(filename):
    cap = cv.VideoCapture(1)

    # size must be defined according to capture (?)
    size = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
    # define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
    # fourcc = cv.VideoWriter_fourcc(*"MP4V")
    # fourcc = cv.VideoWriter_fourcc('a', 'v', 'c', '1')
    out = cv.VideoWriter(f'elements/{filename}_output.mp4', fourcc, 20.0, size)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't recieve frame (stream end?). Exiting...")
            break
        frame = cv.flip(frame, 0)

        # write the flipped frame
        out.write(frame)

        cv.imshow('frame', frame)
        if cv.waitKey(10) == ord('q'):
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    # vid_file_processing(
    # "/Users/williamhbelew/Hacking/ocv_playground/elements/neck_tester_vid.mov")

    vid_file_cap_flip_save('tester1')
