import datetime
import cv2
import datetime as datetime
import imutils
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

from OIP21_lib_ImageProcessing_V6  import *



kernel = np.ones((5, 5), np.uint8)
if __name__ == '__main__':
    img = cv2.imread("./pictures/big_triangles_orginal.png")
    # img = cv2.imread("R001_001.tif")
    # img = cv2.imread("001_002.tif")
    # img = cv2.imread("SingleTCell.png")
    # img=cv2.imread("one_cluster_big_picture.png")
    gray = convert2LIGHT(img[:,:,0],img[:,:,1],img[0:870,:,2])
        # (img[0:870,:,0]+img[0:870,:,1]+img[0:870,:,2])/3
    gauss_img = auto_contrast256(min_filter(gray,2))

    thresh = auto_thresh(gray)
    # thresh2 = auto_thresh(gauss_img)
    # gauss_img, phi, idx, idy = detect_edges(thresh)
    kernel = np.array(np.ones(3), np.uint8)
    # gauss_img, phi, idx, idy = detect_edges(gray)
    # thresh2 =  cv2.erode(cv2.dilate(thresh,kernel),kernel,iterations=1)
    thresh2,phii=laplace_sharpen(thresh)
    # thresh2= (gauss_img)
    cv2.imshow('thresh', thresh)
    cv2.imshow('gauss', gray)
    cv2.imshow('thresh2', thresh2)
    D = ndimage.distance_transform_edt(thresh2)
    localMax = peak_local_max(D, indices=False, min_distance=7,
                              labels=thresh2)
    cv2.imshow("Distance MAp", D)

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh2)

    # plt.imshow("Seperated objects", labels, cmap="spectral")
    print("[INFO] {} unique segments found".format(len(np.unique(labels)) - 1))
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 0:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(thresh2.shape, dtype="uint8")
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
    filename=str("/watershed/"+str(datetime.datetime.utcnow()))
    filename= filename.replace(".","_")
    filename= filename.replace(" ", "")+".png"
    cv2.imwrite("triangles.png", img)
    # plot_image_hist_cumhist(Phi,'Image', 'gray',255,0)
    cv2.waitKey()
