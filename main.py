# v2 - trying to add Tobias segmentation to it 
from numpy import uint8
from OIP21_lib_ImageProcessing_V6  import *
import cv2          # Some of the things in other library took to long
import math
import time


def FloodFillLabeling_modified(imgBIN):

    label = 2
    # collect the non-zero / foreground elements:
    nzi = np.nonzero(imgBIN)
    # make copy:
    IMG = deepcopy(imgBIN)

    zones = []
    # Flood fill loop:
    #for cnt, u in enumerate(FGu):
    for i in np.transpose(nzi):
        IMG , zone = FloodFill_BF_modified(IMG, i[0] , i[1], label)
        if (not zone[0]==0) and (not zone[1]==IMG.shape[0]) and (not zone[2]==0) and (not zone[3]==IMG.shape[1]):
            zones.append(zone)
            label = label + 1
    return IMG ,zones

#insert image, (u,v) (start pixel), label nr
def FloodFill_BF_modified(IMG, u, v, label):
    '''
    Breadth-First Version (we treat lists as queues)
    '''
    xmax=0
    xmin=IMG.shape[0]
    ymax=0
    ymin=IMG.shape[1]
    S = []
    S.append([u,v])
    while S:  # While S is not empty...
        xy = S[0]
        x = xy[0]
        y = xy[1]
        S.pop(0)
        if x <= IMG.shape[0] and y <= IMG.shape[1] and  IMG[x,y] == 1:
            if xmax<x:
                xmax=x
            elif xmin>x:
                xmin=x
            if ymax<y:
                ymax=y
            elif ymin>y:
                ymin=y
            IMG[x,y] = label
            if x+1<IMG.shape[0]:
                S.append([x+1, y])
            if y+1<IMG.shape[1]:
                S.append([x, y+1])
            if y-1>=0:
                S.append([x,y-1])
            if x-1>=0:
                S.append([x-1,y])
    return IMG , [xmax,xmin,ymax,ymin]



img_all_types_big = ['./pictures/big_circles_orginal.tif', './pictures/big_lines_orginal.tif', './pictures/big_triangles_orginal.png']
img_all_types_small = ['./pictures/simp_line_1.PNG','./pictures/simp_triangle_1.PNG', './pictures/simp_circle_1.PNG']
img_triangles = ['./pictures/simp_triangle_1.PNG', './pictures/simp_triangle_2.PNG']
img_circles = ['./pictures/simp_circle_1.PNG','./pictures/simp_circle_2.PNG', './pictures/simp_circle_3.PNG','./pictures/simp_circle_4.PNG' ]
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
tottalNumberOfClusters = 0
tottalNumberOfCircles = 0
tottalNumberOfLines = 0
tottalNumberOfTriangles = 0

tottalTime = 0 
shortestTime = 900000000
longestTime = 0
shortestPicture = 'asd'
longestPicture = 'asd'


for x in img_all_types_big: 

    # Local variables 
    t = time.time()
    clusterArray=[]                     # This should maybe be global? 
    circleArray=[]                      # This should also maybe be global? 
    numberOfClusters = 0
    numberOfCircles = 0
    numberOfLine = 0
    numberOfTriangles = 0

    # Read the image 
    img_orginal = cv2.imread(x, cv2.IMREAD_GRAYSCALE)

    # The Triangle and Circle image have some crap at the bottom we need to cut of, 
    # We might have to look into this a bit more since the crap can also be on the side.. 
    img_orginal = img_orginal[0:870,:] 

    #prepare for region labeling
    img = cv2.medianBlur(img_orginal, 7)
    ret,thresh = cv2.threshold(img,50,255,cv2.THRESH_BINARY)
    kernel = np.ones((7,7), np.uint8)
    threshDil = cv2.dilate(thresh, kernel, iterations=2)

    #255 to 1 since floodfill is expecting that
    threshDilBin = threshDil.copy()
    threshDilBin[threshDilBin == 255] = 1
    threshDilBin = threshDilBin.astype('uint16')

    labelsOIP , zones = FloodFillLabeling_modified(threshDilBin)

    height, width = np.shape(img_orginal)

    for i in zones:
        y2=i[0]
        y1=i[1]
        x2=i[2]
        x1=i[3]
        if(x1>0 and y1>0 and x2<width-1 and y2<height-1):
            clusterArray.append(img_orginal[y1:y2,x1:x2])



    size = math.ceil(math.sqrt(len(clusterArray)))
    count = 1

    for i in clusterArray: 
        numberOfClusters = numberOfClusters + 1
        tottalNumberOfClusters = tottalNumberOfClusters + 1 
        # Filter the cluster and then do edge detect before trying to see 
        # if the cluster is a cluster of circles, lines or hopefully triangles. 

        img_filtered = auto_contrast256(i)      # This does not matter that much for the circles but improves the lines 
        img_filtered = threshold(img_filtered, 80)      # Auto thresholding would prob be better. 
        #ret,thresh = cv2.threshold(img,50,255,cv2.THRESH_BINARY) # Better threshholding? 
        img_filtered, Phi, IDx, IDy= detect_edges(img_filtered, Filter='Prewitt')      


        # Ath the OIP21 library has to be altered so that the data type is Uint8 and not Float64

        # Try to detect circles in the image 
        img_circles = cv2.cvtColor(img_filtered, cv2.COLOR_GRAY2BGR)

        circles = cv2.HoughCircles(img_filtered,                    # HoughCircles only works with unit8 so just typecasting it for simplicity                                # image 
        cv2.HOUGH_GRADIENT,                                         # Method
        1,                                                          # dp inverse resolution (1 = max)
        8,                                                          # minDist, apprximation of the max radius which makes sense
        param1=50,                                                  # Threshold 
        param2=15,                                                  # The lower this is the more false positives and the higher it is it does not detect at all
        minRadius=6,                                                # Minimum Radius 
        maxRadius=12                                                # Maximum Radius
        )

        # The whole drawing lines is not needed and mainly just to make things easier to see.
        try: 
            if circles.any():
                circles = np.uint16(np.around(circles))
                for i in circles[0, :]:
                    center = (i[0], i[1])
                    # circle center
                    cv2.circle(img_circles, center, 1, (0, 100, 100), 3)
                    # circle outline
                    radius = i[2]
                    cv2.circle(img_circles, center, radius, (255, 0, 255), 3)
                    numberOfCircles = numberOfCircles + 1     
                    tottalNumberOfCircles = tottalNumberOfCircles + 1       
        except:
            circleArray.append([0,0,0])     # do we need to keep track of indivitual clusters? 



        # Try to detect lines in the image 
        img_lines = cv2.cvtColor(img_filtered, cv2.COLOR_GRAY2BGR)
        img_lines_prob = np.copy(img_lines)
            
        lines = cv2.HoughLines(img_filtered,                        # Image 
        1,                                                          # Lines 
        np.pi/180,                                                  # Rho 
        40,                                                         # Theta 
        None,                                                       # Srn / Stn 
        40,                                                         # min_Theta
        70)                                                         # Max_Theta
            
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
                numberOfLine = numberOfLine + 1
                tottalNumberOfLines = tottalNumberOfLines + 1
            

        # Triangle stuff 
        triangles = None

        if triangles is not None: 
            numberOfTriangles = numberOfTriangles + 1
            tottalNumberOfTriangles = tottalNumberOfTriangles + 1

        # to show circles in clusters 
        plt.subplot(size,size,count)
        plt.imshow(img_circles,'gray',vmin=0,vmax=255)
        plt.xticks([])
        plt.yticks([])
        count+=1



    # Just gathering some data and stuff, not sure how much is relavant or wanted
    elapse = time.time() - t
    if elapse <= shortestTime:
        shortestTime = elapse
        shortestPicture = x
    
    if elapse >= longestTime:
        longestTime = elapse
        longestPicture = x
    
    tottalTime = tottalTime + elapse

    print("-----------------------------------------------------\n\n")
    print("Currenct picture : ")
    print(x)
    print("Number of Clusters : ")
    print(numberOfClusters)
    print("Number of Circles in clusters : ")
    print(numberOfCircles)
    #for b in circleArray:
        #print(np.size(b)/3)             # Still get 0 as 1 so have to implement this better but this print the number of circles in each cluster
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
print(tottalNumberOfClusters)
print("\nNumber of Circles in clusters : ")
print(tottalNumberOfCircles)
print("Average number of circles in clusters : ")
print(tottalNumberOfCircles/tottalNumberOfClusters)
print("\nNumber of Lines in clusters : ")
print(tottalNumberOfLines)
print("Average number of lines in clusters : ")
print(tottalNumberOfLines/tottalNumberOfClusters)
print("\nNumber of Triangles in clusters : ")
print("Average number of Triangles in clusters : ")
print(tottalNumberOfTriangles/tottalNumberOfClusters)

print("\n\n------------------------ Time Stuff ------------------")
print("Tottal compile time : ")
print(tottalTime)
print("Average compile time : ")
print(tottalTime/len(img_all_types_big))
print("\nShortest compile time : ")
print(shortestTime)
print("Shortest compile time picture : ")
print(shortestPicture)
print("\nLongest compile time : ")
print(longestTime)
print("Longest compile time picture : ")
print(longestPicture)

print("\n\n-----------------------------------------------------")



