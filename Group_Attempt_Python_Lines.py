from OIP21_lib_ImageProcessing_V6  import *
import cv2          # Some of the things in other library took to long
import math

img_orgs = ['simp_line.PNG','simp_triangle.PNG', 'simp_circle.PNG']
img_short = ['simp_triangle.PNG', 'simp_triangle_easy.PNG']
img_circles_many = ['./circles/simp_circle_1.PNG','./circles/simp_circle_2.PNG', './circles/simp_circle_3.PNG','./circles/simp_circle_4.PNG' ]
img_line = ['simp_line.PNG']


attempt_1 = False
attempt_2 = True

for x in img_orgs:
    # Read the image 
    img_orginal = cv2.imread(x, cv2.IMREAD_GRAYSCALE)

    # Filter the image and edge detect it 
    kernel = np.array(N8, np.uint8)
    img_filtered = cv2.morphologyEx(img_orginal, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    img_filtered = auto_contrast256(img_filtered)
    img_filtered = unsharp_mask(img_filtered)
    img_filtered = threshold(img_filtered, 80)
    img_filtered, Phi, IDx, IDy= detect_edges(img_filtered, Filter='Prewitt')

    # Ath the OIP21 library has to be altered so that the data type is Uint8 and not Float64

    # Try to detect circles in the image 
    img_circles = cv2.cvtColor(img_filtered, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(img_filtered.astype(np.uint8),   # HoughCircles only works with unit8 so just typecasting it for simplicity                                # image 
    cv2.HOUGH_GRADIENT,                                         # Method
    1,                                                          # dp inverse resolution (1 = max)
    12,                                                         # minDist, apprximation of the max radius which makes sense
    param1=50,                                                  # Threshold 
    param2=16,                                                  # The lower this is the more false positives and the higher it is it does not detect at all
    minRadius=10,                                               # Minimum Radius 
    maxRadius=16                                                # Maximum Radius
    )
        
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
    except:
        print("------------------ No Circles found ------------------")



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
        
        
    linesP = cv2.HoughLinesP(img_filtered,      # Image  
    1,                                          # Lines 
    np.pi / 180,                                # Rho 
    30,                                         # Theta 
    None,                                       # Threshhold 
    60,                                         # Max Line Length
    60)                                         # Max Line Gap 
        
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(img_lines_prob, (l[0], l[1]), (l[2], l[3]), (0,0,255), 3, cv2.LINE_AA)

    plt.subplot(2,3,1)
    plt.imshow(img_orginal,'gray',vmin=0,vmax=255)
    plt.gca().set_title('Orginal Image')

    plt.subplot(2,3,2)
    plt.imshow(img_filtered,'gray',vmin=0,vmax=255)
    plt.gca().set_title('Filtered Image')

    plt.subplot(2,3,3)
    plt.imshow(img_circles,'gray',vmin=0,vmax=255)
    plt.gca().set_title('Circles Image')

    plt.subplot(2,3,4)
    plt.imshow(img_lines,'gray',vmin=0,vmax=255)
    plt.gca().set_title('Line Image')

   # plt.subplot(2,3,5)
   # plt.imshow(img_lines_prob,'gray',vmin=0,vmax=255)
   # plt.gca().set_title('Probability Lines')

    plt.show()



