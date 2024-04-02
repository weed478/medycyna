import cv2
import numpy as np


def main():
    mog = cv2.createBackgroundSubtractorMOG2()
    cap = cv2.VideoCapture('video2.wmv')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (5, 5), 0)

        fgmask = mog.apply(frame, learningRate=0.01)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        cv2.imshow('frame', frame)
        cv2.imshow('fgmask', fgmask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
