# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 11:52:23 2016

@author: 4chennur
"""

import numpy as np
import cv2
import shutil as copy

f = open('TrashcanBackBB.txt', 'r')
sourceDir = "FirstScenarioDataset/TrashcanBB/"
destDir = "FirstScenarioDataset/trashcanBackgroundclassBB/"
count =0
for line in f:
	abcLine = line.split("\n")[0] + ".png"
	copy.move(sourceDir + abcLine, destDir + abcLine)
	count = count+1