# v2 - trying to add Tobias segmentation to it 
from numpy import uint8
from OIP21_lib_ImageProcessing_V6 import *
import cv2  # Some of the things in other library took to long
import math
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['interactive'] == True

if __name__ == '__main__':
    #img_array = ['./pictures/big_circles_orginal.tif', './pictures/big_lines_orginal.tif', './pictures/big_triangles_orginal.png']
    img_array = ['./pictures/big_lines_orginal.tif']
    # Probably some better way of doing this but just for simplicty a variable or array will be made for each thing
    totalNumberOfClusters = 0  # Region labelling
    totalNumberOfParticles = 0  # particles in cluster: watershed
    totalNumberOfCircles = 0  # from hough transform

    totalTime = 0
    shortestTime = 900000000
    longestTime = 0

    shortestPicture = 'asd'
    longestPicture = 'asd'
    circlePicture = 'asd'
    linePicture = 'asd'
    trianglePicture = 'asd'

    for x in img_array:  ## loop through all the images images stored in a vector
        # Local variables
        t = time.time()
        clusterArray = []  # This should maybe be global?
        circleArray = []  # This should also maybe be global?
        watershed_clusters=[]
        numberOfClusters = 0
        numberOfCircles = 0
        numberOfLine = 0
        numberOfTriangles = 0

        # Read the image
        img_orginal = cv2.imread(x, cv2.IMREAD_GRAYSCALE)

        # The Triangle and Circle image have some stuff at the bottom we need to cut of,
        img_orginal = img_orginal[0:870, :]  ## cut off the bottom manual at this moment

        pre_filterd_image = pre_region_labeling_filtering(img_orginal)

        labelsOIP, zones = FloodFillLabeling_modified(pre_filterd_image)

        clusterArray = segmenting(img_orginal, zones)

        size = math.ceil(math.sqrt(len(clusterArray)))
        count = 1
        count_b = 1

        t = time.time()
        rodsCount = 0
        for i in clusterArray:
            count += 1
            lines, numberOfLines = countRods(i)
            print(numberOfLines)
            rodsCount += numberOfLines
            plt.imshow(lines,'gray',vmin=0,vmax=255)
            plt.show()
        elapse = time.time() - t
        print(rodsCount/len(clusterArray))
        print(elapse)




