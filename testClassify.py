import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import misc as msc
import rospkg
import glob
import preObj as prePro
import detectorDescriptor2 as detDes
import os


def classify(histogram,svm):
    ''' Using a linear SVM for classification '''    
    
    # Get the path to package
#    rospack = rospkg.RosPack()
#    path_to_package = rospack.get_path("object_recognition")
#    
#    svm = cv2.SVM()
#    ## Load the SVM data
#    svm.load(path_to_package + "/Scripts/SVM/svmNoise25000final.dat")
    ## Predict the output label of the test contour 
    prediction = svm.predict(np.float32(histogram))
    return prediction
	
	
	
	
     
    
	
	
	
