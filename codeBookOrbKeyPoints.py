#!/usr/bin/env python

## Importing all the relevant modules 
import numpy as np
import cv2
from matplotlib import pyplot as plt
import preObj as prePro
import detectorDescriptor as detDes
import glob
import os

## Set the path to the input folder and the output folder
rootInputName = "TrainingSet/SVMTrainingSet/"
rootOutputName = "TrainingSet/SVMTrainingORB/"

##Format in which the files should be created
formatName = "*.png"

## List of directories from which the data needs to be collected
listDir = [];
listDir.append("Training/apple/")
listDir.append("Training/background/")
listDir.append("Training/banana/")
listDir.append("Training/cube/")
listDir.append("Training/phone/")

##Initializing the size of the codebook and the number of directories that have been explored so far to collect the data
codeBook = 0
dirCount = 0

## Creating the codebook
for direct in listDir:
	fileList = glob.glob(rootInputName + listDir[dirCount] + formatName)
	print ("Current Directory Name:" + rootInputName + listDir[dirCount])
	count = 0
	for files in fileList:
     ##Read the files from the directory
		inputImage=cv2.imread(fileList[count])
		fileName = os.path.basename(fileList[count])
  
     ## Apply median filetring on the data to remove the noise from the carpet 
		roiImageFiltered = inputImage # cv2.medianBlur(roiImage, 3)
  
     ## Corner detetcor to first get some key points
		kpCorner, roiKeyPointImage = detDes.featureDetectCorner(roiImageFiltered)
  
     ## ORB detector to get the orb features and the descriptor
		kpORB, des, roiKeyPointImage = detDes.featureDescriptorORB(roiImageFiltered, kpCorner)
     ## Assigning the final key points to the variable kp for ease of use
           kp = kpORB 
    ## The if clause would be executed only if some key points are detcted 
		if np.size(kp)>0:
			cv2.imwrite(rootOutputName + listDir[dirCount] + fileName, roiKeyPointImage)
			print ("Path: " +rootOutputName + listDir[dirCount])
			print ("Found some non-zero keypoints for the countours.")
		count = count + 1
		codeBook = codeBook + 1
		
	dirCount = dirCount + 1
						