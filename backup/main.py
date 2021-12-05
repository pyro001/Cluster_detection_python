import matplotlib.pyplot

from OIP21_lib_ImageProcessing_V6 import *
import cv2
import matplotlib.pyplot as plt
import numpy as np

kernel = np.ones((5, 5), np.uint8)
Gauss5 = np.array([[0, 1, 2, 1, 0],
                   [1, 3, 5, 3, 1],
                   [2, 5, 9, 5, 2],
                   [1, 3, 5, 3, 1],
                   [0, 1, 2, 1, 0]])
Gauss5Norm = Gauss5 / np.sum(Gauss5)




if __name__ == '__main__':
    # img = cv2.imread("T001.png")
    img = cv2.imread("R001_001.tif")
    # img = cv2.imread("001_002.tif")

    img=threshold(img[:,:,0],100)
    img_edge, Phi, IDx, IDy= detect_edges(img, Filter='Sobel')
    thinning_img = Thinning(Phi)

    cv2.imshow('Input', img)
    cv2.imshow('thinning',thinning_img)
    cv2.imshow('idx', IDx)
    cv2.imshow('idy', IDy)
    cv2.imshow('phi', Phi)
    cv2.imshow('e', thinning_img)
    # cv2.imshow('blur', canny)

    # matplotlib.pyplot.plot(plot_image_hist_cumhist(img))

    cv2.waitKey(0)
