#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a library gathering all (relevant) image processing functions developed 
within the scope of "Optimisation and Image Processing - OIP" 2018, MCI, SDU.

Put this library in your script folder and load by (for instance):
    
    import OIPlib_ImageProcessing as ip
    
apply functionality by (for instance)

    img, imgRED, imgGREEN, imgBLUE = ip.load_image_GUI()

Created on Tue Sep 11 11:17:33 2018

UPDATE_1, 2018-09-17: 
    ADDED LECTURE 3 CONTENT (LINEAR FILTERING) 
    
UPDATE_2, 2018-09-25:
    ADDED LECTURE 4 CONTENT / FUNCTIONALITY: 
        * NONLINEAR FILTERING     
        * Conversion imgBIN <-> PointSet
        * Basic Operations on PointSets (Union, Intersection, Translation, Reflection)
        
UPDATE_3, 2018-10-01:
    ADDED FUNCTIONS FOR BASIC MORPHOLOGICAL OPERATIONS ON IMAGES AND SETS. 
    
UPDATE_4, 2018-10-10:
    ADDED FUNCTIONS FOR:
        * Edge detection
        * Laplace Sharpening
        * Unsharp Masking
        
        * Thinning (SLOW - faster version yet to come...)
        
        * Hough Line Detection (Original + Accelerated / Vectorised)
       

@author: jost
"""

# ------------------------------------
# IMPORTS: 
# ------------------------------------

# needed almost every time:
import cv2
import imutils
import matplotlib.pyplot as plt  # for plotting
import matplotlib.image as mpimg  # for image handling and plotting
import numpy as np  # for all kinds of (accelerated) matrix / numerical operations
from scipy import ndimage
from scipy.signal import convolve2d
from copy import copy, deepcopy

# tkinter interface module for GUI dialogues (so far only for file opening):
import tkinter as tk
from tkinter.filedialog import askopenfilename

# ------------------------------------
# LOADING, SEPARATING AND CONVERTING TO INTENSITY: 
# ------------------------------------
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def load_image(imgURL):
    ''' This function loads an image '''
    img = mpimg.imread(imgURL)
    img = img * 255
    img = img.astype(np.uint8)
    if len(img.shape) > 2:
        imgRED = img[:, :, 0]
        imgGREEN = img[:, :, 1]
        imgBLUE = img[:, :, 2]
    else:
        print('Single-channel image found! ... returning empty color channels... ')
        imgRED = []
        imgGREEN = []
        imgBLUE = []
    return img, imgRED, imgGREEN, imgBLUE


def load_image_GUI():
    ''' This function loads an image, without a given path to the file, but by
    opening a GUI (based on the tkinter cross-platform interface). Should work on
    most OSs'''

    # GUI-based "getting the URL": 
    root = tk.Tk()
    root.filename = askopenfilename(initialdir="../Images", title="choose your file",
                                    filetypes=(("png files", "*.png"), ("all files", "*.*")))
    print("... opening " + root.filename)
    imgURL = root.filename
    root.withdraw()

    # actually loading the image based on the aquired URL: 
    img = mpimg.imread(imgURL)
    img = img * 255
    img = img.astype(np.uint8)
    imgRED = img[:, :, 0]
    imgGREEN = img[:, :, 1]
    imgBLUE = img[:, :, 2]
    return img, imgRED, imgGREEN, imgBLUE


def crop_levels(imgINT):
    '''helper function to crop all levels back into the 0...255 region'''
    imgINT[imgINT >= 255] = 255
    imgINT[imgINT <= 0] = 0
    return imgINT


# ------------------------------------
# CONVERSION TO 1-CHANNEL INTENSITY: 
# ------------------------------------

def convert2AVG(imgRED, imgGREEN, imgBLUE):
    ''' convert to intensity by averaging '''
    return ((imgRED.astype(np.float) + imgGREEN.astype(np.float) + imgBLUE.astype(np.float)) / 3.0).astype(np.uint8)


def convert2LUM(imgRED, imgGREEN, imgBLUE):
    ''' convert to intensity using the lumosity method (MOST CASES!) '''
    return (0.21 * imgRED + 0.72 * imgGREEN + 0.07 * imgBLUE).astype(np.uint8)


def convert2LIGHT(imgRED, imgGREEN, imgBLUE):
    ''' convert to intensity using the lightness method (Needs to be executed pointwise -> SLOW!) '''
    imgLIGHT = np.zeros(imgRED.shape)
    M, N = imgLIGHT.shape
    for m in range(M):
        for n in range(N):
            imgLIGHT[m, n] = (
                                     np.max([imgRED[m, n], imgGREEN[m, n], imgBLUE[m, n]]).astype(np.float) +
                                     np.min([imgRED[m, n], imgGREEN[m, n], imgBLUE[m, n]]).astype(np.float)
                             ) / 2.0
    return imgLIGHT.astype(np.uint8)


# ------------------------------------
# HISTOGRAM GENERATION: 
# ------------------------------------

def hist256(imgint8):
    ''' manual histogram creation for uint8 coded intensity images'''
    hist = np.zeros(255)
    for cnt in range(255):
        hist[cnt] = np.sum(imgint8 == cnt)
    return (hist)


def cum_hist256(imgint8):
    ''' manual cumulative histogram creation for uint8 coded intensity images'''
    chist = np.zeros(255)
    for cnt in range(255):
        chist[cnt] = np.sum(imgint8 <= cnt)
    return (chist)


# ----------------------------------------------------------------------------
# Point Operations: 
# ----------------------------------------------------------------------------

def threshold(imgINT, ath):
    imgTH = np.zeros(imgINT.shape)
    imgTH[imgINT >= ath] = 255
    return imgTH.astype(np.uint8)


# 0 is black, 255 is white
def threshold2(imgINT, ath):
    return ((imgINT >= ath) * 255).astype(np.uint8)


def threshold_binary(imgINT, ath):
    mask = imgINT >= ath
    imgTH = np.zeros(imgINT.shape)
    imgTH[mask] = 1
    return imgTH.astype(np.uint8)


def threshold_binary2(imgINT, ath):
    return imgINT >= ath


def adjust_brightness(img, a):
    imgB = crop_levels(img.astype(np.float) + a)
    return imgB.astype(np.uint8)


def adjust_contrast(img, a):
    imgC = crop_levels(img.astype(np.float) * a)
    return imgC.astype(np.uint8)


def invert_intensity(img):
    return 255 - img


def auto_contrast256(img):
    alow = np.min(img)
    ahigh = np.max(img)
    amin = 0.
    amax = 255.
    return (amin + (img.astype(np.float) - alow) * (amax - amin) / (ahigh - alow)).astype(np.uint8)


def equalise_histogram256(img):
    M, N = img.shape
    H = cum_hist256(img)
    Hmat = np.zeros(img.shape)
    for cnt in range(255):
        Hmat[img == cnt] = H[cnt]
    imgEqHist = (Hmat * 255 / (M * N)).astype(np.uint8)
    return imgEqHist


def shift_intensities(imgint8, source_int, target_int):
    img_out = imgint8
    img_out[img_out == source_int] = target_int
    return img_out


# ----------------------------------------------------------------------------
# Ostu's  Thresholding:
# ----------------------------------------------------------------------------

def mean_class_variance(hist, thresh):
    wf = 0.0
    uf = 0.0
    ub = 0.0
    wb = 0.0
    total = np.sum(hist)
    # print(total)
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


def auto_thresh(img, mode="Thresholding"):
    hst = hist256(img)
    variance_list = []
    for thresh in range(len(hst)):
        if thresh > 1:
            variance_list.append(mean_class_variance(hst, thresh))
    # print(variance_list)
    # print("MAX tuple: ", max(variance_list))
    var, thresh = max(variance_list)
    # print("MAX tuple: ", thresh)
    if mode == "Thresholding":
        return threshold(img, thresh)
    elif mode == "UpperSave":
        img[img <= thresh] = 0
        return img
    elif mode == "LowerSave":
        img[img >= thresh] = 255
        return img


# ----------------------------------------------------------------------------
# Create Noise: 
# ----------------------------------------------------------------------------

def add_salt_and_pepper(gb, prob):
    '''Adds "Salt & Pepper" noise to an image.
    gb: should be one-channel image with pixels in [0, 1] range
    prob: probability (threshold) that controls level of noise'''

    rnd = np.random.rand(gb.shape[0], gb.shape[1])
    noisy = gb.copy()
    noisy[rnd < prob] = 0
    noisy[rnd > 1 - prob] = 255
    return noisy.astype(np.uint8)


# ----------------------------------------------------------------------------
# FILTER MATRICES:    
# ----------------------------------------------------------------------------

# Define some filter matrices: 
# The 3x3 averaging box:
Av3 = np.array([[1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]]) / 9

# 5x5 Gaussian filter: 
Gauss5 = np.array([[0, 1, 2, 1, 0],
                   [1, 3, 5, 3, 1],
                   [2, 5, 9, 5, 2],
                   [1, 3, 5, 3, 1],
                   [0, 1, 2, 1, 0]])
Gauss5Norm = Gauss5 / np.sum(Gauss5)

# 5x5 Mexican Hat Filter:
Mex5 = np.array([[0, 0, -1, 0, 0],
                 [0, -1, 2, -1, 0],
                 [-1, -2, 16, -2, -1],
                 [0, -1, -2, -1, 0],
                 [0, 0, -1, 0, 0]])
Mex5Norm = Mex5 / np.sum(Mex5)

# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# Laplace Filters: 

# Laplace:
HL4 = np.array([[0., 1., 0.],
                [1., -4., 1.],
                [0., 1., 0.]])

# Laplace 8:
HL8 = np.array([[1., 1., 1.],
                [1., -8., 1.],
                [1., 1., 1.]])

# Laplace 12:
HL12 = np.array([[1., 2., 1.],
                 [2., -12., 2.],
                 [1., 2., 1.]])


# ------------------------------------
# LINEAR FILTERING: 
# ------------------------------------

# ----------------------------------------------------------------------------
# "Naive" / pedestrian Filter implementation (we will rather use the convolutional approach)
def pedestrian_filter(imgINT, H):
    HX, HY = H.shape
    radiusX = np.floor_divide(HX, 2)
    radiusY = np.floor_divide(HY, 2)
    padded = np.pad(imgINT, ((radiusX, radiusX), (radiusY, radiusY)), 'reflect').astype(np.float)
    N, M = imgINT.shape
    filtered = imgINT.astype(np.float)
    for cntN in range(N):
        for cntM in range(M):
            pidN = cntN + radiusX
            pidM = cntM + radiusY
            TMP = padded[pidN - radiusX:pidN + radiusX + 1, pidM - radiusY:pidM + radiusY + 1]
            filtered[cntN, cntM] = np.sum(np.multiply(TMP, H))

    return (crop_levels(filtered)).astype(np.uint8)


# ----------------------------------------------------------------------------
# Linear filtering by convolution
def conv2(x, y, mode='same'):
    # mimic matlab's conv2 function: 
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)


def filter_image(I, H):
    # Convolution-based filtering: 
    Filtered = conv2(np.double(I), np.double(H));
    # Reducing to original size and converting back to uint8: 
    # and CUT to the range between 0 and 255.
    return (crop_levels(Filtered)).astype(np.uint8)


def filter_image_float(I, H):
    # Convolution-based filtering: 
    return conv2(np.double(I), np.double(H))


# ----------------------------------------------------------------------------
# Gaussian Filter Matrix of arbitrary size: 
def create_gaussian_filter(fsize, sigma):
    '''
    Create a Gaussian Filter (square) matrix of arbitrary size. fsize needs 
    to be an ODD NUMBER!
    '''
    # find the center point:
    center = np.ceil(fsize / 2)
    # create a "distance" vector (1xfsize matrix) from the center point: 
    tmp = np.arange(1, center, 1)
    tmp = np.concatenate([tmp[::-1], [0], tmp])
    dist = np.zeros((1, tmp.shape[0]))
    dist[0, :] = tmp
    # create two 1D (x- and y-) Gaussian distributions: 
    Hgx = np.exp(-dist ** 2 / (2 * sigma ** 2))
    Hgy = np.transpose(Hgx)
    # build the outer product to get the full filter matrix: 
    HG = np.outer(Hgy, Hgx)
    # ... normalise... 
    SUM = np.sum(HG)
    HG = HG / SUM
    Hgx = Hgx / np.sqrt(SUM)
    Hgy = Hgy / np.sqrt(SUM)
    return HG, Hgx, Hgy, SUM


# ----------------------------------------------------------------------------
# Laplacian of Gaussian Filter matrix of arbitrary size: 
def create_LoG_filter(fsize, sigma):
    '''
    Create a Gaussian Filter (square) matrix of arbitrary size. fsize needs 
    to be an ODD NUMBER!
    '''
    # find the center point:
    center = np.floor(fsize / 2)
    LoG = np.zeros((fsize, fsize))

    # create relative coordinates: 
    for cntN in range(fsize):
        for cntM in range(fsize):
            rad2 = (cntN - center) ** 2 + (cntM - center) ** 2
            LoG[cntN, cntM] = -1. / (sigma ** 4) * (rad2 - 2. * sigma ** 2) * np.exp(-rad2 / (2. * sigma ** 2))
    # SUM = np.sum(LoG)
    return LoG


# ------------------------------------
# NONLINEAR Filtering: 
# ------------------------------------

def max_filter(imgINT, radius):
    '''filter with the maximum (non-linear) filter of given radius'''
    padded = np.pad(imgINT, ((radius, radius), (radius, radius)), 'reflect').astype(np.float)
    N, M = imgINT.shape
    filtered = np.zeros(imgINT.shape)
    for cntN in range(N):
        for cntM in range(M):
            pidN = cntN + radius
            pidM = cntM + radius
            filtered[cntN, cntM] = np.amax(padded[pidN - radius:pidN + radius + 1, pidM - radius:pidM + radius + 1])
    return filtered.astype(np.uint8)


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


def calc_median(Hmat):
    '''calculate the median of an array...
    This is rather for "educational purposes", since numpy.median is available 
    and potentially (about 10x) faster. '''
    # Hvecsort = sorted([x for x in H.flat]) # LIST SORTING... (EVEN SLOWER!!!)
    Hvecsort = np.sort(Hmat.flat)  # NUMPY VECTOR SORTING...
    length = len(Hvecsort)
    if (length % 2 == 0):
        # if length is even, mitigate: 
        median = (Hvecsort[(length) // 2] + Hvecsort[(length) // 2 - 1]) / 2
    else:
        # otherwise just pick the center element: 
        median = Hvecsort[(length - 1) // 2]
    return median


def median_filter(imgINT, radius):
    padded = np.pad(imgINT, ((radius, radius), (radius, radius)), 'reflect').astype(np.float)
    N, M = imgINT.shape
    filtered = np.zeros(imgINT.shape)
    for cntN in range(N):
        for cntM in range(M):
            pidN = cntN + radius
            pidM = cntM + radius
            filtered[cntN, cntM] = np.floor(
                np.median(padded[pidN - radius:pidN + radius + 1, pidM - radius:pidM + radius + 1]))
    return filtered.astype(np.uint8)


# ------------------------------------
# ------------------------------------
# Morphological Filtering: 
# ------------------------------------
# ------------------------------------

# ------------------------------------
# CONVERSION BETWEEN BINARY IMAGES AND POINT SETS: 

def PointSet_to_imgBIN(coordinates, imgShape):
    '''
    converting point sets to binary images of shape imgShape:
        INPUTS: 
            coordinates:    List of (2D) coordinates, e.g. [[0,0],[1,0],[2,2]]
            imgShape:       image shape, e.g. <matrix>.shape()
        OUTPUT: 
            image:          boolean matrix (with ones at the given coordinates), 
                            wish given shape imgShape
    '''
    cmat = np.matrix(coordinates)
    image = (np.zeros(imgShape)).astype(np.bool)

    for i in range(len(coordinates)):
        if (cmat[i, 0] in range(imgShape[0])) and (cmat[i, 1] in range(imgShape[1])):
            image[cmat[i, 0], cmat[i, 1]] = 1

    return image


def imgBIN_to_PointSet(imgBIN):
    '''
    converting binary images to point sets:
        INPUTS: 
            imgBIN:         boolean matrix 
        OUTPUT: 
            List of (2D) coordinates, e.g. [[0,0],[1,0],[2,2]]
    '''
    return (np.argwhere(imgBIN)).tolist()


# ----------------------------------
# Point Set Operations: 

def PL_union(listone, listtwo):
    ''' calculate the union of two point lists'''
    # convert to set of tuples: 
    l1_set = set(tuple(x) for x in listone)
    l2_set = set(tuple(x) for x in listtwo)
    # return the union as list of lists.
    return [list(x) for x in l1_set | l2_set]


def PL_intersect(listone, listtwo):
    ''' calculate the intersection of two point lists'''
    # convert to tuples:
    l1_set = set(tuple(x) for x in listone)
    l2_set = set(tuple(x) for x in listtwo)
    # return the intersection as list of lists: 
    return [list(x) for x in l1_set & l2_set]


def PL_translate(PointList, Point):
    ''' shift / translate a point list by a point'''
    return [([sum(x) for x in zip(Point, y)]) for y in PointList]


def PL_mirror(PointList):
    ''' mirror / reflect a point list'''
    return [list(x) for x in list((-1) * np.array(PointList))]


# ------------------------------------------------------
# MORPHOLOGICAL OPERATIONS ON BINARY IMAGES
# ------------------------------------------------------

def PL_dilate(PointList_I, PointList_H):
    DILATED_PointList = []
    for q in PointList_H:
        DILATED_PointList = PL_union(DILATED_PointList, PL_translate(PointList_I, q))

    return DILATED_PointList


def img_dilate(I, PLH):
    PLI = imgBIN_to_PointSet(I)
    return PointSet_to_imgBIN(PL_dilate(PLI, PLH), I.shape)


def img_erode(I, PLH):
    PLHFlip = PL_mirror(PLH)
    PLIinv = imgBIN_to_PointSet(1 - I)
    PLI_Erode = PL_dilate(PLIinv, PLHFlip)
    return 1 - PointSet_to_imgBIN(PLI_Erode, I.shape)


def PL_erode(PointList_I, PointList_H):
    # Up  for you to try :-) 
    return


def img_BoundaryExtract(I, PLH):  # PUT YOUR CODE HERE!
    Ierode_inv = 1 - img_erode(I, PLH)
    return PointSet_to_imgBIN(PL_intersect(imgBIN_to_PointSet(I), imgBIN_to_PointSet(Ierode_inv)), I.shape)


def img_open(I, PLH):
    # PUT YOUR CODE HERE!
    return img_dilate(img_erode(I, PLH), PLH)


def img_close(I, PLH):
    # PUT YOUR CODE HERE!
    return img_erode(img_dilate(I, PLH), PLH)


# ------------------------------------
# COMMON STRUCTURING ELEMENTS (POINT SETS)

N4 = [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]]

N8 = [[0, 0], [-1, 0], [1, 0],
      [0, -1], [0, 1], [-1, 1],
      [1, -1], [1, 1], [-1, -1]]

SmallDisk = [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1], [-1, 1], [1, -1], [1, 1], [-1, -1],
             [-2, -1], [-2, 0], [-2, 1],
             [2, -1], [2, 0], [2, 1],
             [-1, -2], [0, -2], [1, -2],
             [-1, 2], [0, 2], [1, 2]]


# ----------------------------------------------------------------------------
# Edge Detection and Sharpening
# ----------------------------------------------------------------------------

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


def laplace_sharpen(imgINT, w=0.1, Filter='L4', Threshold=False, TVal=0.1):
    HL = {
        'L4': HL4,
        'L8': HL8,
        'L12': HL12,
    }.get(Filter, HL4)
    edges = filter_image_float(imgINT.astype(np.float32), HL.astype(np.float32))
    edges = np.divide(edges, np.amax(np.abs(edges)))
    if Threshold:
        edges[np.abs(edges) <= TVal] = 0.0
    filtered = (crop_levels(imgINT.astype(np.float) - w * edges.astype(np.float))).astype(np.uint8)
    return filtered, edges


def unsharp_mask(imgINT, a=4, sigma=2.0, tc=120.):
    # create a gaussian filter:
    fsize = (np.ceil(5 * sigma) // 2 * 2 + 1)  # (5 * next odd integer!)
    HG, HGx, HGy, SUM = create_gaussian_filter(fsize, sigma)

    # filter the image with the Gaussian: 
    imgGaussXY = filter_image_float(filter_image_float(imgINT, HGx), HGy)
    M = imgINT.astype(np.float) - imgGaussXY

    # Create an Edge Map: 
    E, Phi, IDX, IDY = detect_edges(imgINT)

    # Threshold our mask with the Edgemap: 
    M[abs(E) <= tc] = 0.0

    return (crop_levels(imgINT + a * M.astype(np.float))).astype(np.uint8)


# ----------------------------------------------------------------------------
# Thinning
# ----------------------------------------------------------------------------

def neighbours(x, y, image):
    "Return 8-neighbours of image point P1(x,y), in a clockwise order"
    img = image
    x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1
    return [img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],  # P2,P3,P4,P5
            img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1]]  # P6,P7,P8,P9


def transitions(neighbours):
    "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"
    n = neighbours + neighbours[0:1]  # P2, P3, ... , P8, P9, P2
    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)


def Thinning(image):
    "the Zhang-Suen Thinning Algorithm"
    Image_Thinned = image.copy()  # deepcopy to protect the original image
    changing1 = changing2 = 1  # the points to be removed (set as 0)
    while changing1 or changing2:  # iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        rows, columns = Image_Thinned.shape  # x for rows, y for columns
        for x in range(1, rows - 1):  # No. of  rows
            for y in range(1, columns - 1):  # No. of columns
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1 and  # Condition 0: Point P1 in the object regions
                        2 <= sum(n) <= 6 and  # Condition 1: 2<= N(P1) <= 6
                        transitions(n) == 1 and  # Condition 2: S(P1)=1
                        P2 * P4 * P6 == 0 and  # Condition 3
                        P4 * P6 * P8 == 0):  # Condition 4
                    changing1.append((x, y))
        for x, y in changing1:
            Image_Thinned[x][y] = 0
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1 and  # Condition 0
                        2 <= sum(n) <= 6 and  # Condition 1
                        transitions(n) == 1 and  # Condition 2
                        P2 * P4 * P8 == 0 and  # Condition 3
                        P2 * P6 * P8 == 0):  # Condition 4
                    changing2.append((x, y))
        for x, y in changing2:
            Image_Thinned[x][y] = 0
    return Image_Thinned


# ----------------------------------------------------------------------------
# HOUGH TRANSFORM 
# ----------------------------------------------------------------------------

# ------------------------------------
def plot_line_rth(M, N, r, theta, ax):
    """
    Plots a line with (r, theta) parametrisation over a given image.
    """
    uc, vc = np.floor_divide(M, 2), np.floor_divide(N, 2)
    u = np.arange(M)
    v = np.arange(N)
    ax.axis([0, M, 0, N])
    if theta == 0 or theta == np.pi:
        ax.axvline(x=r + uc)
    #        print('zero theta line')
    else:
        x = u - uc
        # y = v-vc        
        # m = np.tan(np.pi/2.-theta)
        m = -np.tan(np.pi / 2 - theta)
        k = r / np.sin(theta)
        ax.plot(x + uc, -(m * x + k) + vc)

    # ------------------------------------


def largest_indices(ARRAY, n):
    """
    Returns the n index combinations referring to the 
    largest values in a numpy array.
    """
    flat = ARRAY.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ARRAY.shape)


# ------------------------------------
def hough_lines_loops(imgBIN, Nth, Nr, K):
    '''
    Computes the Hough transform to detect straight lines in the binary 
    image I (of size M × N ), using Nθ , Nr discrete steps for the angle 
    and radius, respectively. Returns the list of parameter pairs ⟨θi,ri⟩ 
    for the K strongest lines found.
    --- direct implementation of our pseudo code, with if / loops -> SLOW! ---
    '''
    # Find image center: 
    N, M = imgBIN.shape
    uc, vc = np.floor_divide(M, 2), np.floor_divide(N, 2)  # maybe just divide???

    # initialise increments:
    rmax = np.sqrt(uc * uc + vc * vc)
    dth = np.pi / Nth
    dr = 2. * rmax / Nr

    # Create the accumulator Array: 
    Acc = np.zeros((Nth, Nr))

    # Fill the Accumulator Array:
    for u in range(M):
        for v in range(N):
            if imgBIN[v, u] == 1:
                x, y = u - uc, v - vc
                for ith in range(Nth):
                    theta = dth * ith
                    r = x * np.cos(theta) + y * np.sin(theta)
                    ir = np.min([(np.floor_divide(Nr, 2) + np.floor_divide(r, dr)).astype(np.integer), Nr - 1])
                    Acc[ith, ir] = Acc[ith, ir] + 1

    MaxIDX = largest_indices(Acc, K)
    MaxTH = dth * MaxIDX[0][:]
    MaxR = dr * MaxIDX[1][:] - rmax

    return Acc, MaxIDX, MaxTH, MaxR


def hough_lines(imgBIN, Nth, Nr, K):
    '''
    Computes the Hough transform to detect straight lines in the binary 
    image I (of size M × N ), using Nθ , Nr discrete steps for the angle 
    and radius, respectively. Returns the list of parameter pairs ⟨θi,ri⟩ 
    for the K strongest lines found.
    --- Almost all loops avoided / replaced by array arithmetic 
    -> around 100x faster that the "direct" implementation ---
    '''
    # Find image center: 
    N, M = imgBIN.shape
    uc, vc = np.floor_divide(M, 2), np.floor_divide(N, 2)  # maybe just divide???

    # initialise increments:
    rmax = np.sqrt(uc * uc + vc * vc)
    dth = np.pi / Nth
    dr = 2. * rmax / Nr

    # Create the accumulator Array: 
    Acc = np.zeros((Nth, Nr))

    # Fill the Accumulator Array:
    # we can avoid the if statement by numpy's "nonzero" function:
    nzi = np.nonzero(imgBIN)
    y = nzi[0][:] - vc
    x = nzi[1][:] - uc

    # we can avoid the theta / r loop(s) by 
    # .... 1.) an outer product:
    ith = range(Nth)
    theta = dth * ith
    r = np.outer(x, np.cos(theta)) + np.outer(y, np.sin(theta))
    ir = (np.floor_divide(Nr, 2) + np.floor_divide(r, dr)).astype(np.integer)
    # (column index = theta index, the rows represent ALL possible r_index values for one theta value)

    # ... and 2.) a column-wise sum of equal radius values: 
    # (I could not find a way to get rid of the "last" remaining loop over r:)
    for cnt in range(Nr):
        Acc[:, cnt] = np.sum(ir == cnt, axis=0)

    MaxIDX = largest_indices(Acc, K)
    MaxTH = dth * MaxIDX[0][:]
    MaxR = dr * MaxIDX[1][:] - rmax

    return Acc, MaxIDX, MaxTH, MaxR


def hough_circles_loops(imgBIN, Nx, Ny, Nr, K=5, rmin=1, rmax=100):
    '''
    Computes the Hough transform to detect circles in the binary 
    image I (of size M × N ), using Nx, Ny, Nr discrete steps for the center coordinates 
    and radius, respectively. Returns the list of parameter triplets (xi, yi, ri) 
    for the K strongest circles found.
    --- direct implementation of our pseudo code, with if / loops 
        -> VEEEEERY SLOW! ---
    '''
    # Initalise the Accumulator Array: 
    Acc = np.zeros((Nx, Ny, Nr))
    r_increment = (rmax - rmin) / Nr
    r = np.arange(rmin, rmax, r_increment)
    xbar = range(Nx)
    ybar = range(Ny)
    # collect the non-zero / foreground elements: 
    nzi = np.nonzero(imgBIN)
    y = nzi[0][:]
    x = nzi[1][:]
    numFGpix = x.shape[0]
    print("# relevant foreground pixels: %i" % (x.shape[0]))

    # Filling the Accumulator Array:       
    for cntp in range(numFGpix):
        print("Testing foreground pixel %i of %i" % (cntp, x.shape[0]))
        for cntx, xi in enumerate(xbar):
            for cnty, yi in enumerate(ybar):
                for cntr, ri in enumerate(r):
                    if (np.square(x[cntp] - xi) + np.square(y[cntp] - yi) - np.square(ri)) < 1e-16:
                        Acc[cntx, cnty, cntr] = Acc[cntx, cnty, cntr] + 1 / ri / 2 / np.pi

    MaxIDX = largest_indices(Acc, K)
    MaxX = MaxIDX[0][:]  # + rmin
    MaxY = MaxIDX[1][:]  # + rmin
    MaxR = r_increment * MaxIDX[2][:]  # + rmin

    return Acc, MaxIDX, MaxX, MaxY, MaxR


# -------------------------------
def hough_circles(imgBIN, Nx, Ny, Nr, K=5, rmin=1, rmax=100):
    '''
    Computes the Hough transform to detect circles in the binary 
    image I (of size M × N ), using Nx, Ny, Nr discrete steps for the center coordinates 
    and radius, respectively. Returns the list of parameter triplets (xi, yi, ri) 
    for the K strongest circles found.
    --- exchanged the loops by a 3D Meshgrid. about 1500 times faster, but still
        slower than OpenCV implementations... ---
    '''
    # Initalise the Accumulator Array: 
    Acc = np.zeros((Ny, Nx, Nr))
    r_increment = (rmax - rmin) / Nr
    r = np.arange(rmin, rmax, r_increment)
    # print(r)
    xbar = np.arange(Nx)
    ybar = np.arange(Ny)
    # collect the non-zero / foreground elements: 
    nzi = np.nonzero(imgBIN)
    nzi = np.nonzero(imgBIN)
    y = nzi[0][:]
    x = nzi[1][:]
    numFGpix = x.shape[0]
    print("Relevant foreground pixels: %i" % (x.shape[0]))

    # Initialise the coordinate meshgrid
    XBAR, YBAR, RBAR = np.meshgrid(xbar, ybar, r)

    # Filling the Accumulator Array:      
    for cntp in range(numFGpix):
        # Perform the logical operation on the whole Meshgrid at once: 
        tmp = (np.square(x[cntp] - XBAR) + np.square(y[cntp] - YBAR) - np.square(RBAR)) < 1e-16
        Acc = Acc + np.divide(tmp, 2 * np.pi * RBAR)

    MaxIDX = largest_indices(Acc, K)
    MaxY = ybar[MaxIDX[0][:]]
    MaxX = xbar[MaxIDX[1][:]]
    MaxR = r[MaxIDX[2][:]]

    return Acc, MaxIDX, MaxX, MaxY, MaxR


# ----------------------------------------------------------------------------
# Region Labeling 
# ----------------------------------------------------------------------------


# --------------- Flood Fill Labeling -------------------------

def FloodFillLabeling(imgBIN, method='BF'):
    '''
    Algorithm 10.1
    Region marking by flood filling. 
    The binary input image I uses the value 0 for background pixels 
    and 1 for foreground pixels. Unmarked foreground pixels are searched for, 
    and then the region to which they belong is filled. 
    Procedure FloodFill() is defined in three different versions: recursive, depth-first and breadth-first.
    '''
    label = 2
    # collect the non-zero / foreground elements: 
    nzi = np.nonzero(imgBIN)
    FGu = nzi[0][:]
    FGv = nzi[1][:]
    print("Relevant foreground pixels: %i" % (FGu.shape[0]))

    # make copy: 
    IMG = deepcopy(imgBIN)

    # Flood fill loop: 
    for cnt, u in enumerate(FGu):
        v = FGv[cnt]
        if (method == 'recursive'):
            IMG = FloodFill_Recursive(IMG, u, v, label)
        elif (method == 'DF'):
            IMG = FloodFill_DF(IMG, u, v, label)
        elif (method == 'BF'):
            IMG = FloodFill_BF(IMG, u, v, label)
        else:
            print('ERROR: Method must be recursive, BF or DF!')
            break
        label = label + 1
    return IMG


def FloodFill_Recursive(IMG, u, v, label):
    '''
    recursive version
    '''
    if u <= IMG.shape[0] and v <= IMG.shape[1] and IMG[u, v] == 1:
        IMG[u, v] = label
        FloodFill_Recursive(IMG, u + 1, v, label)
        FloodFill_Recursive(IMG, u, v + 1, label)
        FloodFill_Recursive(IMG, u, v - 1, label)
        FloodFill_Recursive(IMG, u - 1, v, label)
    return IMG


def FloodFill_DF(IMG, u, v, label):
    '''
    Depth-First Version (We treat lists as stacks)
    '''
    S = []
    S.insert(0, [u, v])  #
    while S:  # While S is not empty...
        xy = S[0]
        x = xy[0]
        y = xy[1]
        S.pop(0)
        if x <= IMG.shape[0] and y <= IMG.shape[1] and IMG[x, y] == 1:
            IMG[x, y] = label
            S.insert(0, [x + 1, y])
            S.insert(0, [x, y + 1])
            S.insert(0, [x, y - 1])
            S.insert(0, [x - 1, y])
    return IMG


def FloodFill_BF(IMG, u, v, label):
    '''
    Breadth-First Version (we treat lists as queues)
    '''
    S = []
    S.append([u, v])
    while S:  # While S is not empty...
        xy = S[0]
        x = xy[0]
        y = xy[1]
        S.pop(0)
        if x <= IMG.shape[0] and y <= IMG.shape[1] and IMG[x, y] == 1:
            IMG[x, y] = label
            S.append([x + 1, y])
            S.append([x, y + 1])
            S.append([x, y - 1])
            S.append([x - 1, y])
    return IMG


# --------- Sequential Labeling -------------

def SequentialLabeling(imgBIN):
    '''
    Alg. 10.2: Sequential region labeling. 
    The binary input image I uses the value I(u, v) = 0 for background pixels 
    and I(u, v) = 1 for foreground (region) pixels. 
    The resulting labels have the values 2, . . . , label − 1.
    '''
    # make copy: 
    I = deepcopy(imgBIN)

    # STEP 1 - ASSIGN INITIAL LABELS: 
    M, N = imgBIN.shape
    label = 2
    R = []  # Create the sequence of 1-element lists on the fly...
    C = []  # List of label collisions
    for v in range(N):
        for u in range(M):
            if I[u, v] == 1:
                NVals = GetNeighbors8(I, u, v)
                if NVals:
                    Num_N_Labeled = sum(i > 1 for i in NVals)  # number of neighbors with a label > 1
                    Nmax = np.amax(NVals)  # get the maximum label among the neighbors
                    if (Nmax == 0):
                        I[u, v] = label
                        R.append([label])
                        label = label + 1
                    elif (Num_N_Labeled == 1):
                        I[u, v] = Nmax  # set value to the (only) value > 1
                    elif (Num_N_Labeled > 1):
                        I[u, v] = Nmax  # set value to the heighest value > 1
                        for Nval in NVals:
                            if (Nval < Nmax) and (Nval > 1):
                                C.append([Nval, Nmax])  # register collision!
    # >>> THE IMAGE NOW CONTAINS LABELS 0, 2, ..., label-1
    # print(C)
    # print(R)
    # STEP 2 - RESOLVE LABEL COLLISIONS: 
    for collision in C:
        # print('collision:', collision)
        A = collision[0]
        B = collision[1]
        for idx, lbl in enumerate(R):
            # print('labels in R:', lbl)
            if A in lbl:
                idxA = idx
            if B in lbl:
                idxB = idx
        # print('idxA:', idxA)
        # print('idxB:', idxB)
        if idxA != idxB:
            R[idxA].extend(R[idxB])  # merge the two lists into one
            R.pop(idxB)  # remove the other one from the list of lists
            # print('new R:', R)
    # >>> NOW ALL EQUIVALENT LABELS ARE IN THE SAME LIST... 
    # print(R)

    # STEP 3 - Relabeling the image: 
    for u in range(M):
        for v in range(N):
            if (I[u, v] > 1):  # If pixel is labeled:
                for lbl in R:
                    if I[u, v] in lbl:  # Find set of equivalent labels
                        I[u, v] = np.amin(lbl)  # set label to minimum equivalent label
    print(len(R))
    return I


def GetNeighbors4(I, u, v):
    neighborvals = []
    if u > 0:
        neighborvals.append(I[u - 1, v])
    if v > 0:
        neighborvals.append(I[u, v - 1])
    return neighborvals


def GetNeighbors8(I, u, v):
    neighborvals = []
    if (u > 0) and (v > 0):
        neighborvals.append(I[u - 1, v])
        neighborvals.append(I[u - 1, v - 1])
        neighborvals.append(I[u, v - 1])
        neighborvals.append(I[u + 1, v - 1])
    else:
        if u > 0:  # then v == 0
            neighborvals.append(I[u - 1, v])
        if v > 0:  # then u == 0
            neighborvals.append(I[u, v - 1])
            neighborvals.append(I[u + 1, v - 1])
    return neighborvals


# ------------------------------------
# DISTANCE TRANSFORM: 
# ------------------------------------

def DistanceTransform(imgBIN, norm='L2'):
    '''
    Distance transform according to Burger, Burge 2016, Alg. 23.2
    Input: I, a, binary image; norm ∈ {"L1", "L2"}, distance function. 
    Returns the distance transform of I.
    '''
    # STEP 1 - INIT: 
    m1 = 1
    if norm == 'L1':
        m2 = 2
    elif norm == 'L2':
        m2 = np.sqrt(2)
    M, N = imgBIN.shape
    # Create a map D: M x N -> R
    D = (1 - imgBIN).astype(np.float)
    D[D == 0] = np.inf

    # STEP 2 - L -> R PASS: 
    for v in range(N):
        for u in range(M):
            if D[u, v] > 0:
                d1, d2, d3, d4 = np.inf, np.inf, np.inf, np.inf
                if u > 0:
                    d1 = m1 + D[u - 1, v]
                    if v > 0:
                        d2 = m2 + D[u - 1, v - 1]
                if v > 0:
                    d3 = m1 + D[u, v - 1]
                    if u < M - 1:
                        d4 = m2 + D[u + 1, v - 1]
                D[u, v] = np.amin([D[u, v], d1, d2, d3, d4])

    # STEP 3 - R -> L PASS: 
    for v in reversed(range(N)):
        for u in reversed(range(M)):
            if D[u, v] > 0:
                d1, d2, d3, d4 = np.inf, np.inf, np.inf, np.inf
                if u < M - 1:
                    d1 = m1 + D[u + 1, v]
                    if v < N - 1:
                        d2 = m2 + D[u + 1, v + 1]
                if v < N - 1:
                    d3 = m1 + D[u, v + 1]
                    if u > 0:
                        d4 = m2 + D[u - 1, v + 1]
                D[u, v] = np.amin([D[u, v], d1, d2, d3, d4])

    return (D)


# ------------------------------------
# DISTANCE TRANSFORM: 
# ------------------------------------

def LocalMinima(img):
    '''
    find local minia coordinates in an intensity image... 
    IDEA: For calculating local min/max values you can do a little trick.
         You need to perform dilate/erode operation and then compare pixel value with values of original image. 
         If value of original image and dilated/eroded image are equal therefore this pixel is local min/max.
    '''
    PLH = N8  # SmallDisk
    imgDILATE = img_dilate(img, PLH)
    return imgDILATE == img


def LocalMaxima(img):
    '''
    find local minia coordinates in an intensity image... 
    IDEA: For calculating local min/max values you can do a little trick.
         You need to perform dilate/erode operation and then compare pixel value with values of original image. 
         If value of original image and dilated/eroded image are equal therefore this pixel is local min/max.
    '''
    PLH = N8  # SmallDisk
    imgERODE = img_erode(img, PLH)
    return imgERODE == img


# ------------------------------------
# ------------------------------------
# PLOTTING: 
# ------------------------------------
# ------------------------------------

## SIMPLE: 

def plot_image(I, title='Intensity Image', cmap='gray', vmax=255, vmin=0):
    ''' plot the intensity image: '''
    fig, ax = plt.subplots()
    plti = ax.imshow(I, cmap=cmap, vmax=vmax, vmin=vmin)
    ax.set_title(title)
    fig.colorbar(plti, ax=ax)
    return fig, ax


def plot_hist(I, title='Histogram', color='tab:blue'):
    ''' manual histogram plot (for uint8 coded intensity images) '''
    fig, ax = plt.subplots()
    ax.set_xlabel('intensity level')
    ax.set_ylabel('number of pixels', color=color)
    plth = ax.stem(hist256(I), color, markerfmt=' ', basefmt=' ')
    ax.set_title(title)
    return fig, ax, plth


def plot_cumhist(I, title='Cummulative Histogram', color='tab:red'):
    ''' manual cumulative histogram plot (for uint8 coded intensity images) '''
    fig, ax = plt.subplots()
    ax.set_xlabel('intensity level')
    ax.set_ylabel('cumulative n.o.p.', color=color)
    pltch = ax.plot(cum_hist256(I), color=color, linestyle=' ', marker='.')
    ax.set_title(title)
    return fig, ax, pltch


# COMBINED PLOTTING:

def plot_image_hist_cumhist(I, title='Intensity Image', cmap='gray', vmax=255, vmin=0):
    ''' function for the combined plotting of the intensity image, its histogram
    and the cumulative histogram. The histograms are wrapped in a single plot, 
    but since the scales are different, we introduce two y axes (left and right, 
    with different color).'''

    # plot the intensity image: 
    fig, (ax1, ax2) = plt.subplots(2, 1)
    plti = ax1.imshow(I, cmap=cmap, vmax=vmax, vmin=vmin)
    ax1.set_title(title)
    fig.colorbar(plti, ax=ax1)

    # plot the histograms in a yy plot: 
    color = 'tab:blue'
    ax2.set_xlabel('intensity level')
    ax2.set_ylabel('number of pixels', color=color)
    plth = ax2.stem(hist256(I), color, markerfmt=' ', basefmt=' ')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_title('Histogram')

    ax3 = ax2.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    pltch = ax3.plot(cum_hist256(I), color=color, linestyle=' ', marker='.')
    ax3.set_ylabel('cumulative n.o.p.', color=color)
    ax3.tick_params(axis='y', labelcolor=color)

    plt.tight_layout()

    return fig, ax1, ax2, ax3, plti, plth, pltch


def plot_image_all_hists(img, title='Combined Histograms', cmap="gray", vmax=255, vmin=0):
    ''' combined function for plotting the intensity image next to
    * all component histograms, 
    * the intensity histogram and the 
    * cumulative histogram,
    all wrapped in a single plot'''

    # separate and combine
    imgRED = img[:, :, 0]
    imgGREEN = img[:, :, 1]
    imgBLUE = img[:, :, 2]
    imgLUM = convert2LUM(imgRED, imgGREEN, imgBLUE)

    # plot the intensity image: 
    fig, (ax1, ax2) = plt.subplots(2, 1)
    plti = ax1.imshow(imgLUM, cmap=cmap, vmax=vmax, vmin=vmin)
    ax1.set_title(title)
    fig.colorbar(plti, ax=ax1)

    # plot the histograms in a yy plot: 
    color = 'black'
    ax2.set_xlabel('intensity level')
    ax2.set_ylabel('number of pixels', color=color)
    numBins = 256
    pltLUM = ax2.hist(imgLUM.flatten(), numBins, color='black')
    pltR = ax2.hist(imgRED.flatten(), numBins, color='tab:red', alpha=0.5)
    pltG = ax2.hist(imgGREEN.flatten(), numBins, color='tab:green', alpha=0.5)
    pltB = ax2.hist(imgBLUE.flatten(), numBins, color='tab:blue', alpha=0.5)

    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_title('Histogram')

    ax3 = ax2.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    pltch = ax3.plot(cum_hist256(imgLUM), color=color, linestyle=' ', marker='.')
    ax3.set_ylabel('cumulative n.o.p.', color=color)
    ax3.tick_params(axis='y', labelcolor=color)

    plt.tight_layout()

    return fig, ax1, ax2, ax3, plti, pltLUM, pltch


def plot_image_sequence(sequence, title='Intensity Image', cmap='gray', vmax=255, vmin=0):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(sequence.shape[0]):
        ax.imshow(sequence[i, :, :], cmap=cmap, vmax=vmax, vmin=vmin)
        ax.set_title('threshold level: %3i' % (i))
        plt.pause(0.01)
    return fig, ax


def unpad(dens, pad):
    """
    Input:  dens   -- np.ndarray(shape=(nx,ny,nz))
            pad    -- np.array(px,py,pz)

    Output: pdens -- np.ndarray(shape=(nx-px,ny-py,nz-pz))
    """

    nx, ny = dens.shape
    pdens= dens[2:nx-pad,2:ny-pad]
    # pl[2]:nz-pr[2]]

    return pdens


def locwatershed(img, thresh2, padw=4):
    # kernel = np.ones((3, 3), np.uint8)
    # thresh2 = np.pad(thresh2, ((padw, padw), (padw, padw)), 'constant')

    # Using cv2.erode() method
    # thresh2 = cv2.erode(thresh2, kernel, iterations=1)

    # print(img.shape)
    D = ndimage.distance_transform_edt(thresh2)
    localMax = peak_local_max(D, indices=False, min_distance=5,
                              labels=thresh2)
    # cv2.imshow("Distance MAp", D)

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=thresh2)

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
    # cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    # show the output image
    return (img, len(np.unique(labels)) - 1)
