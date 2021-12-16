# v2 - trying to add Tobias segmentation to it 
from matplotlib.pyplot import subplot
from numpy import uint8
from OIP21_lib_ImageProcessing_V6 import *
import cv2  # Some of the things in other library took to long
import math
import time

def filter_lines(MaxTH, MaxR, th_range, r_range):
    add_line = True
    THout = []
    Rout = []
    THout.append(MaxTH[0])
    Rout.append(MaxR[0])
    for i in range(1, len(MaxTH)):
        for j in range(len(THout)):
            if (MaxTH[i] < (THout[j] + th_range) and MaxTH[i] > (THout[j] - th_range)
                and MaxR[i] < (Rout[j] + r_range) and MaxR[i] > (Rout[j] - r_range)):
                add_line = False
        if add_line:
            THout.append(MaxTH[i])
            Rout.append(MaxR[i])
        add_line = True
        
    return THout, Rout

if __name__ == '__main__':
    img_array = ['./pictures/big_circles_orginal.tif', './pictures/big_lines_orginal.tif', './pictures/big_triangles_orginal.png']

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

    count_clusters = 0
    count_foreground_pixels = [[], [], []]
    count_circles = [[], [], []]
    count_rods = [[], [], []]
    count_triangles = [[], [], []]



    for x in img_array:  ## loop through all the images images stored in a vector
        # Local variables
        t = time.time()
        clusterArray = []  # This should maybe be global?
        circleArray = []  # This should also maybe be global?
        watershed_clusters=[]
        numberOfClusters = 0
        

        # Read the image
        img_orginal = cv2.imread(x, cv2.IMREAD_GRAYSCALE)

        # The Triangle and Circle image have some stuff at the bottom we need to cut of,
        img_orginal = img_orginal[0:870, :]  ## cut off the bottom manual at this moment

        pre_filterd_image = pre_region_labeling_filtering(img_orginal)

        labelsOIP, zones = FloodFillLabeling_modified(pre_filterd_image)

        clusterArray = segmenting(img_orginal, zones)
        count_clusters += len(clusterArray)
        output_axs = []
        size = math.ceil(math.sqrt(len(clusterArray)))
        count = 1

        for i in clusterArray:
            numberOfCircles = 0
            numberOfLine = 0
            numberOfTriangles = 0

            numberOfClusters = numberOfClusters + 1
            totalNumberOfClusters = totalNumberOfClusters + 1

            img_edge, img_thresh = pre_conditioning(i)

            watershed_img, c = locwatershed(cv2.cvtColor(i, cv2.COLOR_GRAY2BGR),img_thresh)
            watershed_clusters.append(c)
            circles = openCv_HoughCircles(img_edge, 12, 6, 12)

            if circles is not None:
                for i in circles[0, :]:
                    numberOfCircles = numberOfCircles + 1


#-------------------------------------






            output_axs.append(plt.subplot(size,size,count))
            N, M = img_edge.shape
            if numberOfCircles < 3: 

                Nth = (np.floor_divide(M,2)).astype(np.uint8) # number of THETA values in the accumulator array
                Nr = (np.floor_divide(N,2)).astype(np.uint8)  # number of R values in the accumulator array
                K = 30


                Acc, MaxIDX, MaxTH, MaxR = hough_lines(img_edge, Nth, Nr, K)



                THfil, Rfil = filter_lines(MaxTH, MaxR, 1, 10)

                if K > len(THfil): K = len(THfil)

                avg_angles = []
                for line in range(K):
                    plot_line_rth(M, N, THfil[line], Rfil[line], output_axs[count-1])
                    #plot_line_rth(M, N, MaxR[line], MaxTH[line], output_axs[count-1])

                    avg_angles.append(np.average(np.abs(MaxTH - MaxTH[line])))

                avg_angle = np.average(avg_angles)
                #avg_angle = np.sum(avg_angles)/K
                print("AVERAGE ANGLE")
                print(avg_angle)

            plt.imshow(img_thresh, cmap=plt.cm.gray, extent = [0,M,0,N])
            plt.xticks([])
            plt.yticks([])
            count += 1

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
                    lineClusters = numberOfClusters
                    # Try to detect lines in the image

                    lines = cv2.HoughLines(img_edge,  # Image
                                        1,  # Lines
                                        np.pi / 180,  # Rho
                                        40,  # Theta
                                        None,  # Srn / Stn
                                        40,  # min_Theta
                                        70)  # Max_Theta

                    if lines is not None:
                        for i in range(0, len(lines)):
                            numberOfLine = numberOfLine + 1

            
            count_foreground_pixels[img_array.index(x)].append((np.count_nonzero(img_thresh)/img_thresh.size)*100)
            count_circles[img_array.index(x)].append(numberOfCircles)
            count_rods[img_array.index(x)].append(numberOfLine)
            count_triangles[img_array.index(x)].append(numberOfTriangles)


            #plt.subplot(size, size, count)
            #plt.imshow(watershed_img, 'gray', vmin=0, vmax=255)
            #plt.xticks([])
            #plt.yticks([])
            #count += 1

        plt.show()

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
    print(numberOfLine)
    print("Average number of lines in clusters : ")
    print(numberOfLine / lineClusters)              

    print("\n\nPicture with most ammount of Triangles : ")
    print(trianglePicture)
    print("Numver of Clusters in picture")
    print(trianglesClusters)
    print("Number of Triangles in clusters : ")
    print(numberOfTriangles)
    print("Average number of Triangles in clusters : ")            
    print(numberOfTriangles / trianglesClusters)

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

    fig, axs1 = plt.subplots(4, 1)
    fig, axs2 = plt.subplots(4, 1)
    fig, axs3 = plt.subplots(4, 1)
    fig, axs4 = plt.subplots(4, 1)

    count_circles = np.array(count_circles)
    count_rods = np.array(count_rods)
    count_triangles = np.array(count_triangles)

    print(count_triangles)
    print(count_rods)

    print("FOREGROUND PIXELS")
    axs1[0].hist(count_foreground_pixels, count_clusters, color=["red", "green", "blue"])
    axs1[1].hist(count_foreground_pixels[0], len(count_foreground_pixels[0]), color="red")
    axs1[2].hist(count_foreground_pixels[1], len(count_foreground_pixels[1]), color="green")
    axs1[3].hist(count_foreground_pixels[2], len(count_foreground_pixels[2]), color="blue")
    

    print("CIRCLES")
    axs2[0].hist(count_circles, count_circles.size, color=["red", "green", "blue"])
    axs2[1].hist(count_circles[0], len(count_circles[0]), color="red")
    axs2[2].hist(count_circles[1], len(count_circles[1]), color="green")
    axs2[3].hist(count_circles[2], len(count_circles[2]), color="blue")
    

    print("RODS")
    axs3[0].hist(count_rods, count_rods.size, color=["red", "green", "blue"])
    axs3[1].hist(count_rods[0], len(count_rods[0]), color="red")
    axs3[2].hist(count_rods[1], len(count_rods[1]), color="green")
    axs3[3].hist(count_rods[2], len(count_rods[2]), color="blue")
    

    print("TRIANGLES")
    axs4[0].hist(count_triangles, count_triangles.size, color=["red", "green", "blue"])
    axs4[1].hist(count_triangles[0], len(count_triangles[0]), color="red")
    axs4[2].hist(count_triangles[1], len(count_triangles[1]), color="green")
    axs4[3].hist(count_triangles[2], len(count_triangles[2]), color="blue")
    plt.show()


    plt.boxplot([num_foreground_pixels, ])
    plt.show()

    plt.violinplot(data)
    plt.show()