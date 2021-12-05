from OIP21_lib_ImageProcessing_V6  import *
import cv2          # Some of the things in other library took to long
import math

img_orgs = ['simp_line.PNG','simp_triangle.PNG', 'simp_circle.PNG']
img_short = ['simp_triangle.PNG']

attempt_1 = False
attempt_2 = True

for x in img_short:

    img_org = cv2.imread(x, 0)

    if attempt_2 == True:
        kernel = np.array(N4, np.uint8)
        img_erod = cv2.erode(img_org, kernel, iterations=2)
        img_test = cv2.dilate(img_erod, kernel, iterations=2)

        plt.subplot(2,3,1)
        plt.imshow(img_org,'gray',vmin=0,vmax=255)
        plt.gca().set_title('img_org')

        plt.subplot(2,3,2)
        plt.imshow(img_test,'gray',vmin=0,vmax=255)
        plt.gca().set_title('img_test')

        plt.show()




    if attempt_1 == True: 
        img_median = median_filter(img_org, 3)                              # 3 seems best atm? 
        img_gauss = pedestrian_filter(img_median, Gauss5Norm)               # seems best atm? 

        img_gauss, PHI = laplace_sharpen(img_gauss)

        #img_edge, Phi, IDx, IDy= detect_edges(img_gauss, Filter='Sobel')
        #img_edge, Phi, IDx, IDy= detect_edges(img_gauss, Filter='ISobel')
        img_edge, Phi, IDx, IDy= detect_edges(img_gauss, Filter='Gradient')
        #img_edge, Phi, IDx, IDy= detect_edges(img_gauss, Filter='Perwitt')
    
        #img_edge, Phi, IDx, IDy= (detect_edges(img_gauss, Filter='Gradient'))
        img_thresh = threshold(img_edge, 11)                               # seems best atm? 
        #kernel = np.ones((3,3), np.uint8)
        kernel = np.array(N4, np.uint8)
        img_erod = cv2.erode(img_thresh, kernel, iterations=1)
        img_test = cv2.dilate(img_erod, kernel, iterations=1)

        plt.subplot(2,3,1)
        plt.imshow(img_org,'gray',vmin=0,vmax=255)
        plt.gca().set_title('img_org')

        plt.subplot(2,3,2)
        plt.imshow(img_median,'gray',vmin=0,vmax=255)
        plt.gca().set_title('img_median')

        plt.subplot(2,3,3)
        plt.imshow(img_gauss,'gray',vmin=0,vmax=255)
        plt.gca().set_title('img_gauss')

        plt.subplot(2,3,4)
        plt.imshow(img_edge,'gray',vmin=0,vmax=255)
        plt.gca().set_title('img_edge')

        plt.subplot(2,3,5)
        plt.imshow(img_thresh,'gray',vmin=0,vmax=255)
        plt.gca().set_title('img_thresh')

        plt.subplot(2,3,6)
        plt.imshow(img_test,'gray',vmin=0,vmax=255)
        plt.gca().set_title('img_test')

        plt.show()




#cimg = cv2.cvtColor(img_gauss, cv2.COLOR_GRAY2BGR)

#lines = cv2.HoughLines(img_gauss, 1, np.pi / 180, 150, None, 0, 0)

#if lines is not None:
 #       for i in range(0, len(lines)):
  #          rho = lines[i][0][0]
   #         theta = lines[i][0][1]
    #        a = math.cos(theta)
     #       b = math.sin(theta)
      #      x0 = a * rho
       #     y0 = b * rho
        #    pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
         #   pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
          #  cv2.line(cimg, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)