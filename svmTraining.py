import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import misc as msc
import classifier
import glob
import preObj as prePro
import detectorDescriptor as detDes
import os


svmTrainDataPath ="SVM/TrainingData25000.npy"
svmTrainLabelPath = "SVM/TrainingLabels25000.npy"

svmTrainData = np.load(svmTrainDataPath)
svmTrainLabels = np.load(svmTrainLabelPath)

svm = cv2.SVM()
svm_params = dict( kernel_type = cv2.SVM_RBF, svm_type = cv2.SVM_C_SVC, C=2.67, gamma=5.383 )
svm.train(svmTrainData, svmTrainLabels, params = svm_params)
results = svm.predict_all(svmTrainData)
computeAccuracy = results==svmTrainLabels
print ("Accuracy of the SVM on Training Data is: ", np.float32(np.count_nonzero(computeAccuracy))/np.float32(np.size(computeAccuracy,0))*100)

svm.save("SVM/svmNoise25000final.dat")
