# v2 - trying to add Tobias segmentation to it
import numpy as np
import scipy
from numpy import uint8
from scipy import interpolate
from scipy.optimize import curve_fit

from OIP21_lib_ImageProcessing_V6 import *
import cv2  # Some of the things in other library took to long
import math
import time


if __name__ == '__main__':
    # img_array = ['./pictures/big_circles_orginal.tif', './pictures/big_lines_orginal.tif', './pictures/T001.png']
    #img_array= ['./pictures/big_lines_orginal.tif']
    img_array = 'big_triangles_orginal.png'
    # Probably some better way of doing this but just for simplicty a variable or array will be made for each thing
    totalNumberOfClusters = 0  # Region labelling
    totalNumberOfParticles = 0  # particles in cluster: watershed
    totalNumberOfCircles = 0  # from hough transform

    totalTime = 0
    shortestTime = 900000000
    longestTime = 0
    circleClusters = 0

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
        ForegBackg=[]
        rodArrayCount = [] 
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

        for i in clusterArray:
            numberOfClusters = numberOfClusters + 1
            totalNumberOfClusters = totalNumberOfClusters + 1

            img_edge, img_thresh = pre_conditioning(i)
            img_thresh2,watershed_img, c = locwatershed(cv2.cvtColor(i, cv2.COLOR_GRAY2BGR),img_thresh)
            watershed_clusters.append(c)
            m,n= np.shape(img_thresh)
            try:
                ForegBackg.append(cv2.countNonZero(img_thresh)/(c) )
            except ZeroDivisionError:
                ForegBackg.append(cv2.countNonZero(img_thresh))
            circles = openCv_HoughCircles(img_edge, 12, 6, 12)

            if circles is not None:
                for j in circles[0, :]:
                    numberOfCircles = numberOfCircles + 1

            plt.subplot(2*size, size, count)
            plt.imshow(watershed_img, 'gray', vmin=0, vmax=255)
            plt.xticks([])
            plt.yticks([])
            count += 1
            plt.subplot(2 * size,  size, count)
            plt.imshow(img_thresh2, 'gray', vmin=0, vmax=255)
            plt.xticks([])
            plt.yticks([])
            count += 1
            #-------------------------------------
            # N, M = img_edge.shape
            # if numberOfCircles < 3: 

            #     Nth = (np.floor_divide(M,2)).astype(np.uint8) # number of THETA values in the accumulator array
            #     Nr = (np.floor_divide(N,2)).astype(np.uint8)  # number of R values in the accumulator array
            #     K = 30


            #     Acc, MaxIDX, MaxTH, MaxR = hough_lines(img_edge, Nth, Nr, K)



            #     #MaxTH, MaxR = filter_lines(MaxTH, MaxR, 1, 10)

            #     if K > len(MaxTH): K = len(MaxTH)

            #     avg_angles = []
            #     for line in range(K):
            #         #oip.plot_line_rth(E, MaxTH[i], MaxR[i], ax)
            #         #plot_line_rth(M, N, MaxR[line], MaxTH[line], output_axs[count-1])

            #         avg_angles.append(np.average(np.abs(MaxTH - MaxTH[line])))

            #     avg_angle = np.average(avg_angles)
            #     #avg_angle = np.sum(avg_angles)/K
            #     print("AVERAGE ANGLE")
            #     print(avg_angle)
            # --------------------------------

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
                    trianglePicture = x
                    trianglesClusters = numberOfClusters
                else : 
                    linePicture = x
                    ##momentarily commented // this is annoyingly slow
                    # # Try to detect lines in the image
                    img_lines, numberOfLines = countRods(i)
                    rodArrayCount.append(numberOfLines)



            # plt.subplot(size, size, count)
            # plt.imshow(watershed_img, 'gray', vmin=0, vmax=255)
            # plt.xticks([])
            # plt.yticks([])
            # count += 1

        # Just gathering some data and stuff, not sure how much is relavant or wanted
        x = watershed_clusters
        y=ForegBackg ##normalize the data?
        plt.show()
        #plot rods count histogram
        n, bins, patches = plt.hist(rodArrayCount,20, facecolor='blue', alpha=0.5)
        print("n", n,"bins", bins, "patches", patches)
        plt.xlabel('Bins')
        plt.ylabel('Frequency')
        plt.title('Rods')
        plt.show()
        ##the output looks wierd just take a look
        plt.show()
        n, bins, patches = plt.hist(x,20, facecolor='blue', alpha=0.5)
        print("n", n,"bins", bins, "patches", patches)
        plt.xlabel('Bins')
        plt.ylabel('Frequency')
        plt.show()
        # num_bins = int(np.ceil(max(y) / 20))
        n, bins, patches = plt.hist(y,10,  facecolor='red', alpha=0.5)
        print("n", n, "bins", bins, "patches", patches)
        plt.xlabel('Bins')
        plt.ylabel('Frequency')
        plt.show()
        x = np.linspace(0,450, len(n))
        xdata = np.linspace(0, 450, 40)
        fittingFunction, cov = scipy.stats.distributions.norm.fit(y)
        fitted_data = scipy.stats.distributions.norm.pdf(xdata, fittingFunction, cov)
        plt.plot(xdata, fitted_data, 'r-')
        #curve_fit(f=gaussian, xdata=x, ydata=n)
        # Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
        plt.show()
        plt.scatter(x, n)
        # plt.scatter(xdata,))
        plt.show()


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

        print("Mean size of a cluster: ", round(np.mean(watershed_clusters),2),
        " Median size of a cluster: ", round(np.median(watershed_clusters),2), 
        " Standard deviation of cluster size: ",round(np.std(watershed_clusters),2),
        " Variance of cluster size: ",round(np.var(watershed_clusters),2))
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
    if circleClusters != 0:          
        print(totalNumberOfCircles / circleClusters)
    
    print("\n\nPicture with most ammout of lines")
    print(linePicture)
    print("Number of Clusters in picture")
    # print(lineClusters) ## not defined
    print("Number of Lines in picture : ")
    # print(numberOfLine)
    print("Average number of lines in clusters : ")
    # print(numberOfLine / lineClusters)

    print("\n\nPicture with most ammount of Triangles : ")
    print(trianglePicture)
    print("Numver of Clusters in picture")
    # print(trianglesClusters)
    print("Number of Triangles in clusters : ")
    # print(numberOfTriangles)
    print("Average number of Triangles in clusters : ")            
    # print(numberOfTriangles / trianglesClusters)

    print("Mean number of a cluster particles: ", round(np.mean(watershed_clusters),2),
    " Median number of a cluster particles: ", round(np.median(watershed_clusters),2),
    " Standard Deviation of cluster particles: ",round(np.std(watershed_clusters),2),
    " Variance of cluster particles: ",round(np.var(watershed_clusters),2))

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
