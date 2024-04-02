# AGH UST Medical Informatics 03.2021
# Lab 2 : Segmentation

import cv2 as cv
import numpy as np


if __name__ == '__main__':
    image = cv.imread('abdomen.png')
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    def mouse_callback(event, x, y, flags, params):
        if event ==1:
            img = cv.GaussianBlur(image, (5, 5), 0)
            edges = cv.Canny(img, 40, 140, L2gradient=True)
            edges = cv.morphologyEx(edges, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)))
            cv.imshow('edges', edges)

            segmentation = np.zeros((img.shape[0], img.shape[1]), np.uint8)
            mask = cv.copyMakeBorder(edges, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=0)
            cv.floodFill(segmentation, mask, (x + 1, y + 1), 255)
            cv.imshow('segmentation', segmentation)

            result = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
            result[segmentation != 0] = (0, 0, 255)
            cv.imshow('result', result)

    cv.imshow('image', image)
    cv.setMouseCallback('image', mouse_callback)
    cv.waitKey()
    cv.destroyAllWindows()
