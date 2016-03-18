from skimage.transform import pyramid_gaussian
import cv2

image = cv2.imread("path to file")
for (i, layer) in enumerate(pyramid_gaussian(image, downscale=2)):
	if layer.shape[0] < 50 or layer.shape[1] < 50:
		break
