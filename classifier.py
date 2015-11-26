import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import misc as msc

def kmeans(features,K):
    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(features, K, None, criteria, 10, flags)
    return compactness, labels, centers
    
