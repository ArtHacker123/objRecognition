import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy import misc as msc
import classifier

codeBookPath = "CodeBook/withNoise.npy"

#KMeans code from the classifier
codeBookNoise = np.float32(np.load(codeBookPath))

compactNoise, labelsNoise, centersNoise = classifier.kmeansCodeBook(codeBookNoise, 25000)
print ("Creating codebook from the noise data")

np.save("CodeBook/codeBookNoiseLabels25000.npy", labelsNoise)
np.save("CodeBook/codeBookNoiseCenters25000.npy", centersNoise)

print ("kmneans data saved")
