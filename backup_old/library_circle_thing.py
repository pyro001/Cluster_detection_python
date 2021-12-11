import matplotlib.pyplot as plt
import numpy as np
import time
from OIP21_lib_ImageProcessing_V6  import *
import matplotlib.image as mpimg
import cv2 

def detect_circles(img, ax):
    #def hough_circles(imgBIN, Nx, Ny, Nr, K = 5, rmin = 1, rmax = 100):
    #return Acc, MaxIDX, MaxX, MaxY, MaxR

    N, M = img.shape

    Nx = (np.floor_divide(M,1)).astype(np.uint8)
    Ny = (np.floor_divide(N,1)).astype(np.uint8)
    Nr = (np.floor_divide(M,1)).astype(np.uint8)
    K = 10

    circleArray = hough_circles(img, Nx, Ny, Nr, K=K, rmin = 10, rmax = 30)
    MaxX = circleArray[2]
    MaxY = circleArray[3]
    MaxR = circleArray[4]
    print(MaxX)

    if K > len(MaxX): K = len(MaxX)

    #circles = np.uint16(np.around(circles))
    for i in range(K):
        center = (MaxX[i], MaxY[i])
        radius = MaxR[i]
        draw_circle = plt.Circle(center, radius, fill=False, edgecolor="blue")
        ax.add_artist(draw_circle)

    return circleArray


img = './pictures/SingleTCell.png'

img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

print(img.shape)

#img_filtered = auto_contrast256(img)
#img_filtered = unsharp_mask(img_filtered)
#img_filtered = threshold(img_filtered, 80)
#img_filtered, Phi, IDx, IDy= detect_edges(img_filtered, Filter='Prewitt')
#img_filtered = Thinning(img_filtered)

imgBIN = threshold(img,80)
#plot_image(imgBIN, title='Original Binary Image (Lumosity)', vmax = 1, vmin = 0)
imgBIN, Phi, IDX, IDY = detect_edges(imgBIN, Filter='ISobel')
imgBIN = threshold_binary(imgBIN*255,120)
#plot_image(imgBIN, title='Original Binary Image (Lumosity)', vmax = 1, vmin = 0)
imgBIN = Thinning(imgBIN)
#plot_image(imgBIN, title='Original Binary Image (Lumosity)', vmax = 1, vmin = 0)

N, M = img.shape
K = 5
rmin = 30
rmax = 35
Nr = (rmax-rmin)+1
        
t = time.time()

#Acc, MaxIDX, MaxX, MaxY, MaxR  = hough_circles(img_filtered, 
Acc, MaxIDX, MaxX, MaxY, MaxR  = hough_circles(imgBIN, 
M, 
N, 
Nr, 
K, 
rmin, 
rmax
)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,20))
ax1.imshow(imgBIN, cmap=plt.cm.gray, extent = [0,M,0,N])
plt.gca().invert_yaxis()
for cnt in range(K):
    circle = plt.Circle((MaxY[cnt], N-MaxX[cnt]), MaxR[cnt], color='g', fill=False)
    ax1.add_artist(circle)
ax1.set_title('Original image + detected circles.')


elapse = time.time() - t
print(elapse)
#plt.subplot(2,3,4)
#plt.imshow(img,'gray',vmin=0,vmax=255)
#plt.gca().set_title('Orginal Image')


plt.show()