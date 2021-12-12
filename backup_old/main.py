# v2 - trying to add Tobias segmentation to it 
from numpy import uint8
from OIP21_lib_ImageProcessing_V6 import *
import cv2  # Some of the things in other library took to long
import math
import time

if __name__ == '__main__':
    img_array = ['./pictures/big_circles_orginal.tif', './pictures/big_lines_orginal.tif', './pictures/big_triangles_orginal.png']

    # Probably some better way of doing this but just for simplicty a variable or array will be made for each thing
    totalNumberOfClusters = 0  # Region labelling
    totalNumberOfParticles = 0  # particles in cluster: watershed
    totalNumberOfCircles = 0  # from hough transform
    totalNumberOfLines = 0  # hough lines
    totalNumberOfTriangles = 0  # hough lines triangle detections ?

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

        for i in clusterArray:
            numberOfClusters = numberOfClusters + 1
            totalNumberOfClusters = totalNumberOfClusters + 1

            img_edge, img_thresh = pre_conditioning(i)

            watershed_img, c = locwatershed(cv2.cvtColor(i, cv2.COLOR_GRAY2BGR),img_thresh)
            watershed_clusters.append(c)
            circles = openCv_HoughCircles(img_edge, 12, 6, 12)
            
            # The whole drawing lines is not needed and mainly just to make things easier to see.
            if circles is not None:
                for i in circles[0, :]:
                    numberOfCircles = numberOfCircles + 1


      
            # Principal component analasys 
            if (numberOfCircles/np.sum(watershed_clusters)) >= 0.9: 
                #print("This is probably a circle!")
                totalNumberOfCircles = numberOfCircles
                circlePicture = x
                circleClusters = numberOfClusters
            else :
                if (numberOfCircles/np.sum(watershed_clusters)) >= 0.40: 
                    #print("This is probably a Triangle!")
                    numberOfTriangles = numberOfTriangles + c
                    totalNumberOfTriangles = totalNumberOfTriangles + c
                    trianglePicture = x
                    trianglesClusters = numberOfClusters
                else : 
                    linePicture = x
                    lineClusters = numberOfClusters
                    #print("This is probably a Rod!")
                    # ------------------------------------
                    # ------------------------------------
                    # line Detection : Hough lines
                    # ------------------------------------
                    # ------------------------------------

                    # Try to detect lines in the image
                    img_lines = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2BGR)
                    img_lines_prob = np.copy(img_lines)

                    lines = cv2.HoughLines(img_edge,  # Image
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


            #plt.subplot(size, size, count)
            #plt.imshow(watershed_img, 'gray', vmin=0, vmax=255)
            #plt.xticks([])
            #plt.yticks([])
            #count += 1

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
        print("Particles vs Circles ratio : ")
        print(numberOfCircles/np.sum(watershed_clusters))
        print("Number of Circles in clusters with Hough Line detect : ")
        print(numberOfCircles)
        print("Number of particles in clusters with watersheading : ")
        print(np.sum(watershed_clusters))
        print("Number of Lines in clusters : ")
        print(numberOfLine)
        print("Number of Triangles in clusters : ")
        print(numberOfTriangles)

        print("Run time : ")
        print(elapse)
        print("\n\n-----------------------------------------------------")
        #plt.show()

    print("------------------------ Data Stuff ------------------\n\n")
    print("Total number of Clusters : ")
    print(totalNumberOfClusters)
    
    print("\n\nPicture with most ammount of Circles : ")
    print(circlePicture)
    print("Number of Clusters in picture")
    print(circleClusters)
    print("Number of Circles in picture : ")
    print(totalNumberOfCircles)
    print("Average number of circles in clusters : ")              
    print(totalNumberOfCircles / circleClusters)
    
    print("\n\nPicture with most ammout of lines")
    print(linePicture)
    print("Number of Clusters in picture")
    print(lineClusters)
    print("Number of Lines in picture : ")
    print(totalNumberOfLines)
    print("Average number of lines in clusters : ")
    print(totalNumberOfLines / lineClusters)              

    print("\n\nPicture with most ammount of Triangles : ")
    print(trianglePicture)
    print("Numver of Clusters in picture")
    print(trianglesClusters)
    print("Number of Triangles in clusters : ")
    print(totalNumberOfTriangles)
    print("Average number of Triangles in clusters : ")            
    print(totalNumberOfTriangles / trianglesClusters)

    print("\n\n------------------------ Time Stuff ------------------")
    print("Tottal run time : ")
    print(totalTime)
    print("Average run time : ")
    print(totalTime / len(img_array))
    print("\nShortest run time : ")
    print(shortestTime)
    print("Shortest run time picture : ")
    print(shortestPicture)
    print("\nLongest run time : ")
    print(longestTime)
    print("Longest run time picture : ")
    print(longestPicture)

    print("\n\n-----------------------------------------------------")
