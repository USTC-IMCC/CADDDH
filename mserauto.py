
import cv2
import numpy as np

#Create MSER object
mser = cv2.MSER_create(_delta=1,_min_area=1000,_max_area=3500,_max_variation=0.1)

for i in range(1,1939):
	
    #Your image path i-e receipt path
    img = cv2.imread('/media/hai/AIDL-USTC/USTCPro/original/'+str(i)+'.jpeg')
    
    cropimg = img[300:700,250:1150]
   

    #Convert to gray scale
    gray = cv2.cvtColor(cropimg, cv2.COLOR_BGR2GRAY)

    vis = cropimg.copy()

    #detect regions in gray scale image
    regions, _ = mser.detectRegions(gray)

    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    cv2.polylines(vis, hulls, 1, (0, 0, 0),3)
    
    cv2.imwrite('/media/hai/AIDL-USTC/USTCPro/disposed/'+str(i)+'.jpeg',vis)

    cv2.imshow('img', vis)

    cv2.waitKey(1000)
