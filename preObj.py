import numpy as np
import cv2
from matplotlib import pyplot as plt


def otsuBin(imageGrayInput):
    ret, thresh = cv2.threshold(imageGrayInput, 0, 255,
                                cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return thresh


def meanShift(imageInput):
    meanShifted = cv2.pyrMeanShiftFiltering(imageInput, 20, 20)
    return meanShifted


def cannyEdge(imageInput):
    edgeDetection = cv2.Canny(imageInput, 50, 50)
    return edgeDetection


def adapThresh(imageInput):
    threshAdaptive = cv2.adaptiveThreshold(imageInput, 255, 1, 1, 11, 2)
    return threshAdaptive


def contourFind(prepImage):
    contours, hierarchy = cv2.findContours(prepImage, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)																																									
    return contours, hierarchy


def contourFindFull(prepImage):
    contours, hierarchy = cv2.findContours(prepImage, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)																																									
    return contours, hierarchy


def contourDraw(inputImage, prepImage):
    contours, hierarchy = cv2.findContours(prepImage, cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_SIMPLE)
    inputImageCopy = inputImage.copy()
    cv2.drawContours(inputImageCopy, contours, -1, (255, 0, 0), 3)
    return inputImageCopy
