import numpy as np
import cv2
from matplotlib import pyplot as plt
import preObj as prePro
import detectorDescriptor as detDes
import glob
import os

rootInputName = "TrainingSet/SVMTrainingSet/"
rootOutputName = "TrainingSet/SVMTrainingORB/"
formatName = "*.png"

listDir = [];
listDir.append("Training/apple/")
listDir.append("Training/background/")
listDir.append("Training/banana/")
listDir.append("Training/cube/")
listDir.append("Training/phone/")
codeBook = 0
dirCount = 0
for direct in listDir:
	fileList = glob.glob(rootInputName + listDir[dirCount] + formatName)
	print ("Current Directory Name:" + rootInputName + listDir[dirCount])
	count = 0
	for files in fileList:
		inputImage=cv2.imread(fileList[count])
		fileName = os.path.basename(fileList[count])
		roiImageFiltered = inputImage # cv2.medianBlur(roiImage, 3)
		kp, roiKeyPointImage = detDes.featureDetectCorner(roiImageFiltered)
		kp, des, roiKeyPointImage = detDes.featureDescriptorORB(roiImageFiltered, kp)
               
		if np.size(kp)>0:
			cv2.imwrite(rootOutputName + listDir[dirCount] + fileName, roiKeyPointImage)
			print ("Path: " +rootOutputName + listDir[dirCount])
			print ("Found some non-zero keypoints for the countours.")
		count = count + 1
		codeBook = codeBook + 1
		
	dirCount = dirCount + 1
						