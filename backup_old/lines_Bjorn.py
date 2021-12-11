import OIP21_lib_ImageProcessing_V6 as oip
import numpy as np
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
import cv2
import math

def filter_lines(lines, th_range, r_range):
    filtered_lines = []
    for line in lines:

        filtered_lines.append()


imgs = []
img1, r1, g1, b1 = oip.load_image('001_002.tif')
img2, r2, g2, b2 = oip.load_image('simp_line.png')
img3, r3, g3, b3 = oip.load_image('T001.png')
imgs = [img1, img2, img3]
r = [r1, r2, r3]
g = [g1, g2, g3]
b = [b1, b2, b3]




for i in range(len(imgs)):
    #print(imgs[i].shape)
    try: 
        imgs[i] = imgs[i][:,:,0]
    except: 
        pass
    #print(imgs[i].shape)
    #imgs[i] = oip.convert2LUM(r[i], g[i], b[i])
    
    #imgs[i], Phi, IDX, IDY = oip.detect_edges(imgs[i], Filter='ISobel')
    imgs[i] = oip.threshold_binary(imgs[i], 10)

    # imgs[i] = oip.unsharp_mask(imgs[i])
    # imgs[i] = oip.adjust_brightness(imgs[i], -40)
    # imgs[i] = oip.adjust_contrast(imgs[i], 20)
    
    
    # #imgs[i] = oip.median_filter(imgs[i], 4)
    # #imgs[i] = cv2.medianBlur(imgs[i],3)
    

    # #imgs[i] = oip.auto_contrast256(imgs[i])

    # #oip.plot_cumhist(img)
    # print(imgs[i].shape)
    

#filtered, edges = oip.laplace_sharpen(imgs[1])
#E, Phi, IDx, IDy = oip.detect_edges(imgs[1])
E = imgs[1]

#E = oip.threshold(E, 10)
#E = oip.threshold_binary(E, 40)
#print(E)

N, M = E.shape
Nth = (np.floor_divide(M,2)).astype(np.uint8) # number of THETA values in the accumulator array
Nr = (np.floor_divide(N,2)).astype(np.uint8)  # number of R values in the accumulator array
K = 20
Acc, MaxIDX, MaxTH, MaxR = oip.hough_lines(E, Nth, Nr, K)
print(MaxTH)
print(MaxR)


ax = plt.subplot()

filtered_lines = []
for i in range(K):
    #oip.plot_line_rth(E, MaxTH[i], MaxR[i], ax)
    oip.plot_line_rth(M, N, MaxR[i], MaxTH[i], ax)
    # for line in filtered_lines:
    #     for range(-)

#cimg = cv2.cvtColor(E, cv2.COLOR_GRAY2BGR)


#lines = cv2.HoughLines(imgs[1], 10, np.pi / 180, 150, None, 0, 0)


# if lines is not None:
#         for i in range(0, len(lines[0])):
#             rho = lines[1][i]
#             theta = lines[0][i]
#             a = math.cos(theta)
#             b = math.sin(theta)
#             x0 = a * rho
#             y0 = b * rho
#             pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
#             pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
#             cv2.line(cimg, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)
    


#print(np.shape(imgs[1]))
#plt.imshow(E, cmap='gray')
ax.imshow(E, cmap=plt.cm.gray, extent = [0,M,0,N])
#plt.imshow(cimg)
#plot_cumhist(img)

#plt.rcParams['figure.figsize'] = [20, 10]

#plt.imshow(line_plot, cmap='gray')
plt.show()



