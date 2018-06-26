import cv2
import numpy as np

for i in range(1,500):
	
    img_rgb = cv2.imread('/media/hai/AIDL-USTC/USTCPro/original/' + str(i) + '.jpeg')
    img_orinal = cv2.imread('/media/hai/AIDL-USTC/USTCPro/original/' + str(i) + '.jpeg')
    j = 1
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('/home/hai/python_work/match_template/templeclub/004.png',0)
    w, h = template.shape[::-1]
    print(w)
    print(h)
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.6
    a = []
    b = []
    loc = np.where( res >= threshold)

    for pt in zip(*loc[::-1]):
		
          

        if (((pt[0] + pt[0] + w)/2 < 500) and ((pt[1] + pt[1] + h)/2 < 800) and ((pt[0] + pt[0] + w)/2 > 100) and ((pt[1] + pt[1] + h)/2 > 100) ) :
            a.append((pt[0] + pt[0] + w)/2)
            b.append((pt[1] + pt[1] + h)/2)

        
            cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 1)
            print(a[j-1])
            print(b[j-1])
            print(j) 
            j = j + 1

				
    cv2.imshow('result',img_rgb)
    print(sum(a)/j,sum(b)/j)

    imageo = cv2.circle(img_orinal,(int(sum(a)/j),int(sum(b)/j)),1,(255,0,255),3)
    viso = img_orinal.copy()
    cv2.imwrite('/home/hai/python_work/match_template/resultwhirboneleft/' + str(i) + '.jpeg',viso)
    cv2.waitKey(500)

