# v2 - trying to add Tobias segmentation to it 
from numpy import uint8
from OIP21_lib_ImageProcessing_V6 import *
import cv2  # Some of the things in other library took to long
import math
import time


if __name__ == '__main__':

    img_all_types_big = ['./pictures/big_circles_orginal.tif', './pictures/big_lines_orginal.tif', './pictures/big_triangles_orginal.png']
    #img_all_types_big = ['./pictures/big_lines_orginal.tif']# ["./pictures/big_triangles_orginal.png"]#
    img_all_types_small = ['./pictures/simp_line_1.PNG', './pictures/simp_triangle_1.PNG',
                           './pictures/simp_circle_1.PNG']
    img_triangles = ['./pictures/simp_triangle_1.PNG', './pictures/simp_triangle_2.PNG']
    img_circles = ['./pictures/simp_circle_1.PNG', './pictures/simp_circle_2.PNG', './pictures/simp_circle_3.PNG',
                   './pictures/simp_circle_4.PNG']
    img_lines = ['./pictures/simp_line_1.PNG']

    img_just_circle_big = ['./pictures/big_circles_orginal.tif']
    img_one_of_each_cluster_type = ['./pictures/one_cluster_big_picture.png']

    #  ---------------------- Issues --------------------
    # Rods/Lines and Triangles not implemented currently
    #
    # Have to figgure out what the relevant gatherd data is?
    #
    # Figgure out Optimization
    #
    # Filtering might need a tune up depending on implementation of Rods + Triangles and stuff
    #
    # When cutting the image in line 115 I think we should maybe look into it a bit more since the lines
    # would still have the half clusters on the side..

    # Probably some better way of doing this but just for simplicty a variable or array will be made for each thing
    totalNumberOfClusters = 0  # Region labelling
    totalNumberOfCircles = 0  # from hough transform
    totalNumberOfLines = 0  # hough lines
    totalNumberOfTriangles = 0  # hough lines triangle detections ?
    totalNumberOfParticles = 0  # particles in cluster: watershed

    totalTime = 0
    shortestTime = 900000000
    longestTime = 0
    shortestPicture = 'asd'
    longestPicture = 'asd'

    for x in img_all_types_big:  ## loop through all the images images stored in a vector

        # Local variables
        t = time.time()
        clusterArray = []  # This should maybe be global?
        circleArray = []  # This should also maybe be global?
        numberOfClusters = 0
        numberOfCircles = 0
        numberOfLine = 0
        numberOfTriangles = 0

        # Read the image
        img_orginal = cv2.imread(x, cv2.IMREAD_GRAYSCALE)

        # The Triangle and Circle image have some stuff at the bottom we need to cut of,

        img_orginal = img_orginal[0:870, :]  ## cut off the bottom manual at this moment
        # :: automate this// feed it based  on the sample image
        # ------------------------------------
        # ------------------------------------
        # Segmentation:
        # ------------------------------------
        # ------------------------------------

        # prepare for region labeling
        img = cv2.medianBlur(img_orginal, 7)
        thresh = auto_thresh(img)
        kernel = np.ones((7, 7), np.uint8)
        threshDil = cv2.dilate(thresh, kernel, iterations=2)

        # 255 to 1 since floodfill is expecting that
        threshDilBin = threshDil.copy()
        threshDilBin[threshDilBin == 255] = 1
        threshDilBin = threshDilBin.astype('uint16')

        labelsOIP, zones = FloodFillLabeling_modified(threshDilBin)

        height, width = np.shape(img_orginal)
        ## storing the image coords in a vector
        for i in zones:
            y2 = i[0]
            y1 = i[1]
            x2 = i[2]
            x1 = i[3]
            if (x1 > 0 and y1 > 0 and x2 < width - 1 and y2 < height - 1):
                clusterArray.append(img_orginal[y1:y2, x1:x2])## the clusters are now in a vector

        size = math.ceil(math.sqrt(len(clusterArray)))
        count = 1
        ####################################################################
        watershed_clusters=[]
        for i in clusterArray:
            padw=3
            i=np.pad(i, ((padw, padw), (padw, padw)), 'constant')
            numberOfClusters = numberOfClusters + 1
            totalNumberOfClusters = totalNumberOfClusters + 1
            # Filter the cluster and then do edge detect before trying to see
            # if the cluster is a cluster of circles, lines or hopefully triangles.

            img_contrast = auto_contrast256(i)  # This does not matter that much for the circles but improves the lines
            img_thresholded = auto_thresh(img_contrast)  # Auto thresholding would prob be better.
            # ret,thresh = cv2.threshold(img,50,255,cv2.THRESH_BINARY) # Better threshholding?
            watershed_img,c= locwatershed(cv2.cvtColor(i, cv2.COLOR_GRAY2BGR),img_thresholded)
            watershed_clusters.append(c)
            # except Exception as Error:
            #     watershed_img=i
            #     print(Error)
            img_edges, Phi, IDx, IDy = detect_edges(img_thresholded, Filter='Prewitt')

            # Ath the OIP21 library has to be altered so that the data type is Uint8 and not Float64

            # Try to detect circles in the image
            #img_circles = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2BGR)
            #img_circles = cv2.cvtColor(watershed_img, cv2.COLOR_GRAY2BGR)

            circles = cv2.HoughCircles(img_edges,
                                       # HoughCircles only works with unit8 so just typecasting it for simplicity
                                       # image
                                       cv2.HOUGH_GRADIENT,  # Method   /bTODO ::: look at this
                                       1,  # dp inverse resolution (1 = max)/bTODO ::: look at this
                                       8,  # minDist, approximation of the max radius which makes sense
                                       param1=50,  # Threshold
                                       param2=15, # #:: tolerance of the algorithm how many points on the circle the
                                       # algo needs to make an image The lower this is the more false positives and
                                       # the higher it is it does not detect at all
                                       minRadius=6,  # Minimum Radius :: generated Circle radius control
                                       maxRadius=12  # Maximum Radius
                                       )

            # The whole drawing lines is not needed and mainly just to make things easier to see.
            try:
                if circles.any():
                    circles = np.uint16(np.around(circles))
                    currenctCircles = 0
                    for i in circles[0, :]:
                        # painting the circles onto the image
                        center = (i[0], i[1])
                        # circle center
                        cv2.circle(watershed_img, center, 1, (0, 100, 100), 3)
                        # circle outline
                        radius = i[2]
                        cv2.circle(watershed_img, center, radius, (255, 0, 255), 3)
                        # end of image painting
                        numberOfCircles = numberOfCircles + 1
                        totalNumberOfCircles = totalNumberOfCircles + 1
                        currenctCircles = currenctCircles +1
                    print("Hough line thing : ",currenctCircles," Watershed circles : ", c)
            except:
                circleArray.append([0, 0, 0])  # do we need to keep track of indivitual clusters?
      
            # ------------------------------------
            # ------------------------------------
            # line Detection : Hough lines
            # ------------------------------------
            # ------------------------------------

            # Try to detect lines in the image
            img_lines = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2BGR)
            img_lines_prob = np.copy(img_lines)

            lines = cv2.HoughLines(img_edges,  # Image
                                   1,  # Lines
                                   np.pi / 180,  # Rho
                                   40,  # Theta
                                   None,  # Srn / Stn
                                   40,  # min_Theta
                                   70)  # Max_Theta

            if lines is not None:
                for i in range(0, len(lines)):
                    rho = lines[i][0][0]
                    theta = lines[i][0][1]
                    a = math.cos(theta)
                    b = math.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
                    pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
                    cv2.line(img_lines, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
                    numberOfLine = numberOfLine + 1
                    totalNumberOfLines = totalNumberOfLines + 1

            # ------------------------------------
            # ------------------------------------
            # Triangles : ??
            # ------------------------------------
            # ------------------------------------

            # Triangle stuff
            triangles = None

            if triangles is not None:
                numberOfTriangles = numberOfTriangles + 1
                totalNumberOfTriangles = totalNumberOfTriangles + 1

            # to show circles in clusters

            plt.subplot(size, size, count)
            #plt.imshow(img_circles, cmap=plt.cm.nipy_spectral)
            plt.imshow(watershed_img, 'gray', vmin=0, vmax=255)
            plt.xticks([])
            plt.yticks([])
            count += 1

        # Just gathering some data and stuff, not sure how much is relavant or wanted
        elapse = time.time() - t
        if elapse <= shortestTime:
            shortestTime = elapse
            shortestPicture = x

        if elapse >= longestTime:
            longestTime = elapse
            longestPicture = x

        totalTime = totalTime + elapse

        print("-----------------------------------------------------\n\n")
        print("Currenct picture : ")
        print(x)
        print("Number of Clusters : ")
        print(numberOfClusters)
        print("Number of Circles in clusters with Hough Line detect : ")
        print(numberOfCircles)
        print("Number of particles in clusters with watersheading : ")
        #print()
        # for b in circleArray:
        # print(np.size(b)/3)             # Still get 0 as 1 so have to implement this better but this print the number of circles in each cluster
        print("Number of Lines in clusters : ")
        print(numberOfLine)
        print("Number of Triangles in clusters : ")
        print(numberOfTriangles)

        print("Compile time : ")
        print(elapse)
        print("\n\n-----------------------------------------------------")
        plt.show()

    print("------------------------ Data Stuff ------------------\n\n")
    print("Number of Clusters : ")
    print(totalNumberOfClusters)
    print("\nNumber of Circles in clusters : ")
    print(totalNumberOfCircles)
    print("Average number of circles in clusters : ")
    print(totalNumberOfCircles / totalNumberOfClusters)
    print("\nNumber of Lines in clusters : ")
    print(totalNumberOfLines)
    print("Average number of lines in clusters : ")
    print(totalNumberOfLines / totalNumberOfClusters)
    print("\nNumber of Triangles in clusters : ")
    print("Average number of Triangles in clusters : ")
    print(totalNumberOfTriangles / totalNumberOfClusters)

    print("\n\n------------------------ Time Stuff ------------------")
    print("Tottal compile time : ")
    print(totalTime)
    print("Average compile time : ")
    print(totalTime / len(img_all_types_big))
    print("\nShortest compile time : ")
    print(shortestTime)
    print("Shortest compile time picture : ")
    print(shortestPicture)
    print("\nLongest compile time : ")
    print(longestTime)
    print("Longest compile time picture : ")
    print(longestPicture)

    print("\n\n-----------------------------------------------------")
