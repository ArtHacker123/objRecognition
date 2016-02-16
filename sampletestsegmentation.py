# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 19:28:03 2016

@author: 4chennur
"""

import cv2
import testSegmented as tS
frame = cv2.imread('frame0032.jpg')

tS.processAndClassify(frame)