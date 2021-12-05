from Cluster_detection_python.oiplib import *
import cv2          # Some of the things in other library took to long
import math


Number = 2

# Start loading in the 3 image types, 
# The first type is a Circle cluster image
# The second type is a Line/Rod cluster image
# The third type is a triangle cluster image
if Number == 1:
    #img = mpimg.imread(r"001_002.tif",0)
    img = cv2.imread("001_002.tif", 0)
elif Number == 2:
    #img = mpimg.imread(r"R001_001.tif",0)
    img = cv2.imread("R001_001.tif", 0)
elif Number == 3:
    img = cv2.imread(r"T001.png",0)

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






# ret,thresh1 = cv2.threshold(img, 10, 200 ,cv2.THRESH_BINARY)

# kernel = np.ones((8,8), np.uint8)
# img = cv2.erode(img, kernel, iterations=1)
# img = cv2.dilate(img, kernel, iterations=1)




#sharp_image = image - a * Laplacian(image)
img = img - 0.2 * cv2.Laplacian(img, cv2.CV_64F)
img = adjust_brightness(img, -40)
img = adjust_contrast(img, 20)

E, Phi, IDx, IDy = detect_edges(img)
#E = oip.threshold(E, 10)
E = threshold_binary(E, 127)



# Now we make the image a little bit sharper in hopes that the edge detection will be a 
# bit more consistent 

#img = unsharp_mask(img) # sharpening it brigns out all the noise Ahh! 



# Drawing the circles and stuff 

# https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d 
# https://docs.opencv.org/4.x/da/d53/tutorial_py_houghcircles.html

cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

lines = cv2.HoughLines(img, 1, np.pi / 180, 150, None, 0, 0)

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
            cv2.line(cimg, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)

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
plt.imshow(cimg, cmap='gray')
#plt.imshow(cimg)
#plot_cumhist(img)

plt.show()
