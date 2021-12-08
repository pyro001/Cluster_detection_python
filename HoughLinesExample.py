"""
@file hough_lines.py
@brief This program demonstrates line finding with the Hough transform
"""
import sys
import math
import cv2 
import numpy as np
import matplotlib.pyplot as plt
from OIP21_lib_ImageProcessing_V6  import *

 
# Loads an image
src = cv2.imread('simp_line.PNG', cv2.IMREAD_GRAYSCALE)

    
    
#dst = cv2.Canny(src, 50, 200, None, 3)
#dst = src

kernel = np.array(N8, np.uint8)
img_filtered = cv2.morphologyEx(src, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
img_filtered = auto_contrast256(img_filtered)
img_filtered = unsharp_mask(img_filtered)
img_filtered = threshold(img_filtered, 80)
img_filtered, Phi, IDx, IDy= detect_edges(img_filtered, Filter='Prewitt')

# Copy edges to the images that will display the results in BGR
img_lines = cv2.cvtColor(img_filtered, cv2.COLOR_GRAY2BGR)
img_lines_prob = np.copy(img_lines)
    
lines = cv2.HoughLines(img_filtered, 1, np.pi / 180, 150, None, 0, 0)
    
if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(img_lines, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
    
    
linesP = cv2.HoughLinesP(img_filtered, 1, np.pi / 180, 50, None, 50, 10)
    
if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(img_lines_prob, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

plt.subplot(2,3,1)
plt.imshow(src,'gray',vmin=0,vmax=255)
plt.gca().set_title('img_org')

plt.subplot(2,3,2)
plt.imshow(img_lines,'gray',vmin=0,vmax=255)
plt.gca().set_title('img_test')

plt.subplot(2,3,3)
plt.imshow(img_lines_prob,'gray',vmin=0,vmax=255)
plt.gca().set_title('img_opening')

plt.show()
    
