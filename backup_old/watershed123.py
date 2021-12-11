import cv2
import imutils as imutils
import numpy as np
import skimage.segmentation
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
from oiplib import  *
# Load in image, convert to gray scale, and Otsu's threshold
image = cv2.imread('one_cluster_big_picture.png')
# img = cv2.imread("T001.png")
# image= cv2.imread("R001_001.tif")
# image = cv2.imread("001_002.tif")
image= image[0:870,0:1000]
#
# kernel = np.array(N8, np.uint8)
# image2=image[:,:,0]
# img_filtered = cv2.morphologyEx(image2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
# autocon = auto_contrast256(img_filtered)
# unsharp = unsharp_mask(autocon,10,2.0,120)
# # thresh = threshold(unsharp, 80)
# kernel = np.array(N8, np.uint8)
# erode_img = cv2.erode(autocon,kernel,iterations=1)
# # edge, Phi, IDx, IDy= detect_edges(img_filtered, Filter='Prewitt')
# # gray=laplace_sharpen(gray)
# # thresh=cv2.threshold(gray,127,255,np.uint8)
# # thresh = cv2.threshold(E, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#
# # Compute Euclidean distance from every binary pixel
# # to the nearest zero pixel then find peaks
# distance_map = ndimage.distance_transform_edt(equalise_histogram256(erode_img))
# local_max = peak_local_max(distance_map, indices=False, min_distance=5, labels=erode_img)
#
# # Perform connected component analysis then apply Watershed
# markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]
# labels = skimage.segmentation.watershed(-distance_map, markers, mask=erode_img)
#
# # Iterate through unique labels
# total_area = 0
# for label in np.unique(labels):
#     if label == 0:
#         continue
#
#     # Create a mask
#     mask = np.zeros(img_filtered.shape, dtype="uint8")
#     mask[labels == label] = 255
#
#     # Find contours and determine contour area
#     cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = cnts[0] if len(cnts) == 2 else cnts[1]
#     c = max(cnts, key=cv2.contourArea)
#     area = cv2.contourArea(c)
#     total_area += area
#     cv2.drawContours(image, [c], -1, (36,255,12), 4)
#     # cv2.drawContours(thresh, [c], -1, (36, 255, 12), 4)
# print(total_area)
# cv2.imshow('image', image)
# cv2.imshow('thresh', thresh)
# cv2.imshow('Markers', erode_img)
# # cv2.imshow('Labels', labels);
# cv2.waitKey()

shifted= pedestrian_filter(image[:,:,0],H=Gauss5Norm)
gray= shifted
# image[:,:,1:2]=0

# shifted = image
# cv2.pyrMeanShiftFiltering(image, 5, 5)
cv2.imshow("Input", shifted)
# convert the mean shift image to grayscale, then apply
# Otsu's thresholding
# gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Thresh", thresh)
# kernel = np.array(N8, np.uint8)
# erode_img = cv2.erode(autocon,kernel,iterations=1)
# cv2.imshow("erosion", thresh)
# compute the exact Euclidean distance from every binary
# pixel to the nearest zero pixel, then find peaks in this
# distance map
D = ndimage.distance_transform_edt(thresh)
localMax = peak_local_max(D, indices=False, min_distance=10,
	labels=thresh)
cv2.imshow("Distance MAp", D)

# perform a connected component analysis on the local peaks,
# using 8-connectivity, then appy the Watershed algorithm
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=thresh)
print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
for label in np.unique(labels):
	# if the label is zero, we are examining the 'background'
	# so simply ignore it
	if label == 0:
		continue
	# otherwise, allocate memory for the label region and draw
	# it on the mask
	mask = np.zeros(gray.shape, dtype="uint8")
	mask[labels == label] = 255
	# detect contours in the mask and grab the largest one
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	c = max(cnts, key=cv2.contourArea)
	# draw a circle enclosing the object
	((x, y), r) = cv2.minEnclosingCircle(c)
	cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 1)
	# cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
	# 	cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)