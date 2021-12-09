import imutils
import matplotlib.pyplot as plt
from matplotlib import *
from numpy import uint64
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from oiplib import *
import cv2
import numpy as np


def mean_class_variance(hist, thresh):
    wf = 0.0
    uf = 0.0
    ub = 0.0
    wb = 0.0
    total = np.sum(hist)
    # print(total)
    if len(hist) > 256:
        raise IndexError("object too large, recheck input")
    for i in range(len(hist)):
        if i < thresh:
            wb += hist[i]
            ub = ub + (i * hist[i])
        elif i >= thresh:
            wf += hist[i]
            uf = uf + (i * hist[i])
    ub = ub / wb
    uf = uf / wf
    wb = wb / total
    wf = wf / total
    # print(wb,wf,uf,ub,total)
    variance = float(wb * wf*((ub - uf)** 2))
    return (variance, thresh)


def auto_thresh(img):
    hst = hist256(img)
    variance_list = []
    for thresh in range(len(hst)):
        if thresh>0:
            variance_list.append(mean_class_variance(hst, thresh))
    # print(variance_list)
    # print("MAX tuple: ", max(variance_list))
    var,thresh=max(variance_list)
    print("MAX tuple: ", thresh)
    return threshold(img,thresh)

kernel = np.ones((5, 5), np.uint8)
if __name__ == '__main__':
    # img = cv2.imread("T001.png")
    # img = cv2.imread("R001_001.tif")
    # img = cv2.imread("001_002.tif")
    img = cv2.imread("one_cluster_big_picture.png")
    gray = img[:, :, 0]
    thresh=auto_thresh(gray)
    gauss_img=pedestrian_filter(thresh,Gauss5Norm)
    thresh2 = auto_thresh(gauss_img)
    cv2.imshow('thresh', thresh)
    cv2.imshow('gauss', gauss_img)
    cv2.imshow('thresh2', thresh2)
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=5,
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
        cv2.circle(img, (int(x), int(y)), int(r), (0, 255, 0), 1)
    # cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
    # 	cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    # show the output image
    cv2.imshow("Output", img)
    # cv2.imshow('erode', img_close)
    # cv2.imwrite("../modified.png", img_edge2)
    # plot_image_hist_cumhist(Phi,'Image', 'gray',255,0)
    cv2.waitKey()
