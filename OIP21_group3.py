import math
# tkinter interface module for GUI dialogues (so far only for file opening):
import tkinter as tk
from copy import deepcopy
from tkinter.filedialog import askopenfilename

import cv2
import imutils
import matplotlib.image as mpimg  # for image handling and plotting
import matplotlib.pyplot as plt  # for plotting
import numpy as np  # for all kinds of (accelerated) matrix / numerical operations
from numpy import uint8

import scipy
from scipy import ndimage
from scipy.signal import convolve2d
from scipy import interpolate
from scipy.optimize import curve_fit
# ------------------------------------
# LOADING, SEPARATING AND CONVERTING TO INTENSITY:
# ------------------------------------
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

import time

def power_law(x, a, b):
    return a * np.power(x, b)

def gaussian(x, a, b, c):
    return a*np.exp(-np.power(x - b, 2)/(2*np.power(c, 2)))

def auto_thresh(img, mode="Thresholding"):
    hst = hist256(img)
    variance_list = []
    for thresh in range(len(hst)):
        if thresh > 1:
            variance_list.append(mean_class_variance(hst, thresh))
    var, thresh = max(variance_list)
    if mode == "Thresholding":
        return threshold(img, thresh)
    elif mode == "UpperSave":
        img[img <= thresh] = 0
        return img
    elif mode == "LowerSave":
        img[img >= thresh] = 255
        return img

def mean_class_variance(hist, thresh):
    wf = 0.0
    uf = 0.0
    ub = 0.0
    wb = 0.0
    total = np.sum(hist)
    if len(hist) > 256:
        raise IndexError("object image depth too large, recheck input")
    for i in range(len(hist)):
        if i < thresh:
            wb += hist[i]
            ub = ub + (i * hist[i])
        elif i >= thresh:
            wf += hist[i]
            uf = uf + (i * hist[i])
    if wb == 0:
        ub = 0
    else:
        ub = ub / wb
    if wf == 0:
        uf = 0
    else:
        uf = uf / wf
    wb = wb / total
    wf = wf / total

    variance = float(wb * wf * ((ub - uf) ** 2))
    return variance, thresh

def skeletonize(img):
    """ OpenCV function to return a skeletonized version of img, a Mat object"""

    #  hat tip to http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/

    img = img.copy()  # don't clobber original
    skel = img.copy()

    skel[:, :] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
    x = cv2.countNonZero(img)
    count = 1
    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:, :] = eroded[:, :]
        count += 1
        if cv2.countNonZero(img) <= 0 or count >= 10:
            break

    return skel

def countRods(i):
    img = i.copy()
    # scale the image so i have more pixels to play with
    scale_percent = 200  # percent of original size
    width = img.shape[1] * 2
    height = img.shape[0] * 2
    dim = (width, height)
    img = cv2.resize(img, dim)

    # blur to make mask to remove everything outside the cluster
    img_blur = cv2.blur(img, (5, 5))
    ret, mask = cv2.threshold(img_blur, 20, 255, cv2.THRESH_BINARY)
    img_noise = cv2.bitwise_and(img, img, mask=mask)

    # apply mexican hat
    kernel = np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]])
    img_hat1 = cv2.filter2D(img_noise, -1, kernel)
    img_hat2 = cv2.filter2D(img_hat1, -1, kernel)
  
    # theshold
    thresh = auto_thresh(img_hat2)

    #distance transform
    D = ndimage.distance_transform_edt(thresh)
    localMax = peak_local_max(D, indices=False, min_distance=3,labels=thresh)
    localMax = np.multiply(localMax,255).astype('uint8')
    kernel = np.ones((3,3), np.uint8)
    #dilate for easier line finding
    localMax = cv2.dilate(localMax, kernel, iterations=1)
    #find lines and draw on top until no more lines is found
    drawLines = []
    lines = cv2.HoughLines(localMax,1,(1*np.pi)/180,8)
    while lines is not None :
        drawLines.append([lines[0][0][0],lines[0][0][1]])
        a = math.cos(lines[0][0][1])
        b = math.sin(lines[0][0][1])
        x0 = a * lines[0][0][0]
        y0 = b * lines[0][0][0]
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(localMax, pt1, pt2, (0,0,0), 3)
        lines = cv2.HoughLines(localMax,1,(1*np.pi)/180,8)

    #draw the lines on image
    img_lines = img.copy()
    for i in drawLines:
        a = math.cos(i[1])
        b = math.sin(i[1])
        x0 = a * i[0]
        y0 = b * i[0]
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(img_lines, pt1, pt2, (100,0,0), 1)

    return img_lines, len(drawLines)

def FloodFillLabeling_modified(imgBIN):
    label = 2
    # collect the non-zero / foreground elements:
    nzi = np.nonzero(imgBIN)
    # make copy:
    IMG = deepcopy(imgBIN)

    zones = []
    # Flood fill loop:
    # for cnt, u in enumerate(FGu):
    for i in np.transpose(nzi):
        IMG, zone = FloodFill_BF_modified(IMG, i[0], i[1], label)
        if (not zone[0] == 0) and (not zone[1] == IMG.shape[0]) and (not zone[2] == 0) and (
                not zone[3] == IMG.shape[1]):
            zones.append(zone)
            label = label + 1
    return IMG, zones

# insert image, (u,v) (start pixel), label nr
def FloodFill_BF_modified(IMG, u, v, label):
    '''
    Breadth-First Version (we treat lists as queues)
    '''
    xmax = 0
    xmin = IMG.shape[0]
    ymax = 0
    ymin = IMG.shape[1]
    S = []
    S.append([u, v])
    while S:  # While S is not empty...
        xy = S[0]
        x = xy[0]
        y = xy[1]
        S.pop(0)
        if x <= IMG.shape[0] and y <= IMG.shape[1] and IMG[x, y] == 1:
            if xmax < x:
                xmax = x
            elif xmin > x:
                xmin = x
            if ymax < y:
                ymax = y
            elif ymin > y:
                ymin = y
            IMG[x, y] = label
            if x + 1 < IMG.shape[0]:
                S.append([x + 1, y])
            if y + 1 < IMG.shape[1]:
                S.append([x, y + 1])
            if y - 1 >= 0:
                S.append([x, y - 1])
            if x - 1 >= 0:
                S.append([x - 1, y])
    return IMG, [xmax, xmin, ymax, ymin]

def pre_region_labeling_filtering(img):
    # The Triangle and Circle image have some stuff at the bottom we need to cut of,
    # img_orginal = img[0:870, :]  ## cut off the bottom manual at this moment

    # prepare for region labeling
    img_b = cv2.medianBlur(img, 7)
    thresh = auto_thresh(img_b)
    kernel = np.ones((7, 7), np.uint8)
    threshDil = cv2.dilate(thresh, kernel, iterations=2)

    # 255 to 1 since floodfill is expecting that
    threshDilBin = threshDil.copy()
    threshDilBin[threshDilBin == 255] = 1
    threshDilBin = threshDilBin.astype('uint16')

    return threshDilBin

def segmenting(img, zones):
    array = []
    height, width = np.shape(img)
    ## storing the image coords in a vector
    for i in zones:
        y2 = i[0]
        y1 = i[1]
        x2 = i[2]
        x1 = i[3]
        if (x1 > 0 and y1 > 0 and x2 < width - 1 and y2 < height - 1):
            array.append(img[y1:y2, x1:x2])  ## the clusters are now in a vector
    return array

def pre_conditioning(img):
    padw = 3
    i = np.pad(img, ((padw, padw), (padw, padw)), 'constant')

    img_contrast = auto_contrast256(i)  # This does not matter that much for the circles but improves the lines
    img_thresholded = auto_thresh(img_contrast)  # Auto thresholding would prob be better.
    img_edges, Phi, IDx, IDy = detect_edges(img_thresholded, Filter='Prewitt')
    return img_edges, img_thresholded

def openCv_HoughCircles(img, tolerance, minRadius, maxRadius):
    circles = cv2.HoughCircles(img,
                               # HoughCircles only works with unit8 so just typecasting it for simplicity
                               # image
                               cv2.HOUGH_GRADIENT,  # Method   /bTODO ::: look at this
                               1,  # dp inverse resolution (1 = max)/bTODO ::: look at this
                               8,  # minDist, approximation of the max radius which makes sense
                               param1=50,  # Threshold
                               param2=tolerance,
                               # #:: 12 best tolerance of the algorithm how many points on the circle the
                               # algo needs to make an image The lower this is the more false positives and
                               # the higher it is it does not detect at all
                               minRadius=minRadius,  # Minimum Radius :: generated Circle radius control
                               maxRadius=maxRadius  # Maximum Radius
                               )
    return circles

def find_particle(img, x,y,r, percentage=0.1): #set to 10% by default
    np.ceil(r)
    img_temp = img[int(y - r):int(y + r), int(x - r):int(x + r)]
    whitePixles = cv2.countNonZero(img_temp)
    if whitePixles >= (r * r * 4) * percentage:  # Calculate the area of a circle and then multiplies it by the % and then checks if it is bigger then the white in the area
        return True  # if the smallest white is bigger then area*% append it
    else:
        return False

def locwatershed(img_org, thresh2, modifier=1.9):
    zoom = False
    redux = False
    img = img_org.copy()
    l, b = np.shape(thresh2)
    if l < 50 and b < 50:
        zoom = True
        img = cv2.resize(img, (0, 0), fx=1.2, fy=1.2)
        thresh2 = cv2.resize(thresh2, (0, 0), fx=4, fy=4)
    kernel = np.array(Mex5, np.uint8)  
    temp = min_filter(thresh2, 3)
    temp = cv2.dilate(cv2.erode(temp, kernel, iterations=1), kernel, iterations=1)

    if cv2.countNonZero(temp) > 0.4 * cv2.countNonZero(thresh2) and not zoom:
        thresh2 = temp
        redux=True
        #print(cv2.countNonZero(temp), 0.6 * cv2.countNonZero(thresh2))
    else:
        kernel =  np.ones((3, 3), np.uint8)
        thresh2=cv2.erode(thresh2, kernel, iterations=1)
    minareacircles = []
    avgr = []

    D = ndimage.distance_transform_edt(thresh2)
    localMax = peak_local_max(D, indices=False, min_distance=5,
                              labels=thresh2)
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then apply the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh2)

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
        if(find_particle(thresh2,x,y,r,0.3)):
            minareacircles.append([x, y, r])
            avgr.append(r)
    try:
        minareacircles = [i for i in minareacircles if i[2] >= np.mean(avgr)-modifier*np.std(avgr)]

        for l in minareacircles:
            x, y, z = l
            cv2.circle(img, (int(x), int(y)), int(r), (0, 255, 0), 1)
    except Exception as E:
        print("Error: ", E, len(np.unique(labels)))
    if redux:
        print(minareacircles,cv2.countNonZero(temp) ,cv2.countNonZero(thresh2))
    return thresh2, img, len((minareacircles))

# -----------------------------------------------------------------------------------------
# Functions from OIP21_lib_ImageProcessing_V6.py librarry 

# 5x5 Mexican Hat Filter:
Mex5 = np.array([[0, 0, -1, 0, 0],
                 [0, -1, 2, -1, 0],
                 [-1, -2, 16, -2, -1],
                 [0, -1, -2, -1, 0],
                 [0, 0, -1, 0, 0]])

# Edge Detection Filters: 
# Derivative:
HDx = np.array([[-0.5, 0., 0.5]])
HDy = np.transpose(HDx)

# Prewitt:
HPx = np.array([[-1., 0., 1.],
                [-1., 0., 1.],
                [-1., 0., 1.]]) / 6.
HPy = np.transpose(HPx)

# Sobel:
HSx = np.array([[-1., 0., 1.],
                [-2., 0., 2.],
                [-1., 0., 1.]]) / 8.
HSy = np.transpose(HPx)

# Improved Sobel:
HISx = np.array([[-3., 0., 3.],
                 [-10., 0., 10.],
                 [-3., 0., 3.]]) / 32.
HISy = np.transpose(HPx)

def min_filter(imgINT, radius):
    '''filter with the minimum (non-linear) filter of given radius'''
    padded = np.pad(imgINT, ((radius, radius), (radius, radius)), 'reflect')  # .astype(np.float)
    N, M = imgINT.shape
    filtered = np.zeros(imgINT.shape)
    for cntN in range(N):
        for cntM in range(M):
            pidN = cntN + radius
            pidM = cntM + radius
            filtered[cntN, cntM] = np.floor(
                np.amin(padded[pidN - radius:pidN + radius + 1, pidM - radius:pidM + radius + 1]))
    return filtered.astype(np.uint8)

def conv2(x, y, mode='same'):
    # mimic matlab's conv2 function: 
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)

def filter_image_float(I, H):
    # Convolution-based filtering: 
    return conv2(np.double(I), np.double(H))

def auto_contrast256(img):
    alow = np.min(img)
    ahigh = np.max(img)
    amin = 0.
    amax = 255.
    return (amin + (img.astype(np.float) - alow) * (amax - amin) / (ahigh - alow)).astype(np.uint8)

def detect_edges(imgINT, Filter='Sobel'):
    Hx, Hy = {
        'Gradient': [HDx, HDy],
        'Sobel': [HSx, HSy],
        'Prewitt': [HPx, HPy],
        'ISobel': [HISx, HISy]
    }.get(Filter, [HSx, HSy])
    # Filter the image in x- and y-direction 
    IDx = filter_image_float(imgINT, Hx)
    IDy = filter_image_float(imgINT, Hy)
    # create intensity maps and phase maps: 
    # E = (np.sqrt(np.multiply(IDx, IDx)+np.multiply(IDy, IDy))).astype(np.float32)
    E = (np.sqrt(np.multiply(IDx, IDx) + np.multiply(IDy, IDy))).astype(np.uint8)
    Phi = (np.arctan2(IDy, IDx) * 180 / np.pi).astype(np.float32)
    return E, Phi, IDx, IDy

def hist256(imgint8):
    ''' manual histogram creation for uint8 coded intensity images'''
    hist = np.zeros(255)
    for cnt in range(255):
        hist[cnt] = np.sum(imgint8 == cnt)
    return (hist)

def threshold(imgINT, ath):
    imgTH = np.zeros(imgINT.shape)
    imgTH[imgINT >= ath] = 255
    return imgTH.astype(np.uint8)