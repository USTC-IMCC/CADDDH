import cv2
import numpy as np


for j in range(1,1940):

    img = cv2.imread('/media/hai/AIDL-USTC/USTCPro/disposed/' + str(j) + '.jpeg',0)
    ret,thresh = cv2.threshold(img,10,255,0)
    img,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    print('The number of contours to have been found is:')
    print(len(contours))
    print('\n')
    a = len(contours)
    for i in range(0,a):
        cnt = contours[i]
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        center = (int(x),int(y))
        radius = int(radius)
        area = cv2.contourArea(cnt)
        s = 3.14*radius*radius
        l=2*3.14*radius
        perimeter = cv2.arcLength(cnt,True)
        if((radius > 20) and (radius < 40) and ((s-area)<2000) and ((l-perimeter)<60) and ((x < 400) or (x > 600)) and ((y > 100) and (y < 300))):
		
            print('The center of this contours is:')
            print(center) #compute center
            print('\n')
            print('The radius of this contours is:')
            print(radius) #compute radius
            print('\n')
        
            print('The area of this contours is:')
            print(area) #test area
            print('\n')
            print('The area of this outgetcontours is:')
            print(3.14*radius*radius)
            print('\n')
        
            print('The perimeter of this contours is:')
            print(perimeter) #compute perimeter
            print('\n')
            print('The perimeter of this outgetcontours is:')
            print(2*3.14*radius)
            print('\n')
            imga = cv2.circle(img,center,radius,(125,125,125),5)
            vis = img.copy()
            cv2.imshow('image', vis)
            #cv2.waitKey(5)
            #drawpoints
            imgo = cv2.imread('/media/hai/AIDL-USTC/USTCPro/disposed/' + str(j) + '.jpeg')
            imageo = cv2.circle(imgo,(int(x),int(y)),1,(255,0,255),3)
            viso = imgo.copy()
            cv2.imwrite('/media/hai/AIDL-USTC/USTCPro/disposed/' + str(j) + '.jpeg',viso)
        
       
        cv2.waitKey(5)
