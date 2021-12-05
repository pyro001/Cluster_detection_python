from OIP21_lib_ImageProcessing_V6 import *
import cv2          # Some of the things in other library took to long

Number = 1

# Start loading in the 3 image types, 
# The first type is a Circle cluster image
# The second type is a Line/Rod cluster image
# The third type is a triangle cluster image
if Number == 1:
    img = cv2.imread("001_002.tif", 0)
elif Number == 2:
    img = cv2.imread("R001_001.tif", 0)
elif Number == 3:
    img = cv2.imread("T001.png", 0)


# Then we cut the gunk that is not needed from the image, this would be un-needed data
# or things that should not be in the image like half-clusters and other things that would
# screw with the data. 

if Number == 1:                 # these values are found by experimenting
    img = img[0:870,:]
elif Number == 2: 
    img = img[0:870,0:1000]
elif Number == 3:
    print("No Changes needed for this image type right now") 

# Now that we have the trash in the picture types cleaned up we can do stuff like
# Make them the same size? 
# Clean up the static in them?
# Make em awesome? 

# Same size thing here if we run into problems of clean up being less effective between the
# different cluster types and stuff 


# 


#img.astype(np.uint8)

#pedestrian_filter(img, Mex5)
#filter_image(img, Mex5)

# Morpholocial filtering? erroding -> growing, lecture 05 
# The function cv::morphologyEx can perform advanced morphological 
# transformations using an erosion and dilation as basic operations.
#ret,thresh1 = cv2.threshold(img,100,255,cv2.THRESH_BINARY)
#img = cv2.medianBlur(img, 5)


#ret,thresh1 = cv2.threshold(img, 15, 150 ,cv2.THRESH_BINARY)

kernel = np.ones((5,5), np.uint8)
img = cv2.erode(img, kernel, iterations=2)
img = cv2.dilate(img, kernel, iterations=2)



# ATH Min Max filter 

# Now we make the image a little bit sharper in hopes that the edge detection will be a 
# bit more consistent 

#img = unsharp_mask(img) # sharpening it brigns out all the noise Ahh! 



# Drawing the circles and stuff 

# https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d 
# https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html

cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img,                                    # image 
cv2.HOUGH_GRADIENT,                     # Method
1,                                      # dp inverse resolution (1 = max)
8,                                      # minDist, apprximation of the max radius which makes sense
param1=50,                              # Threshold 
param2=7,                               # The lower this is the more false positives and the higher it is it does not detect at all
minRadius=4,                            # Minimum Radius 
maxRadius=9                             # Maximum Radius
)
    
circles = np.uint16(np.around(circles))

for i in circles[0, :]:
    center = (i[0], i[1])
    # circle center
    cv2.circle(cimg, center, 1, (0, 100, 100), 3)
    # circle outline
    radius = i[2]
    cv2.circle(cimg, center, radius, (255, 0, 255), 3)

print(np.shape(img))
plt.imshow(img, cmap='gray')
plt.imshow(cimg)
#plot_cumhist(img)

plt.show()
