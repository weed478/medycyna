# AGH UST Medical Informatics 03.2021
# Lab 2 : Segmentation

import cv2 as cv
import numpy as np


if __name__ == '__main__':
    image = cv.imread('abdomen.png')
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    def mouse_callback(event, x, y, flags, params):
        if event ==1:
            img = cv.medianBlur(image, 5)
            
            selected_value = img[y, x]
            diff = np.abs(img.astype(np.int32) - selected_value).astype(np.uint8)

            cv.imshow("diff", diff)

            segmentation = cv.inRange(diff, 0, 10)

            cv.imshow('segmentation', segmentation)

            mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)
            cv.floodFill(segmentation, mask, (x, y), 255)
            mask = mask[1:-1, 1:-1] * 255

            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))

            cv.imshow('mask', mask)

    cv.imshow('image', image)
    cv.setMouseCallback('image', mouse_callback)
    cv.waitKey()
    cv.destroyAllWindows()
