import cv2
import os
import numpy as np



def otherfour(category):
    for y in os.listdir(r'./testdoctor/'+str(category)+'/'):
        n = y.split('.')
        img = cv2.imread('./testdoctor/'+str(category)+'/' + str(n[0]) + '.jpg')
        ma = []
        j=0
        for q in os.listdir(r'./temptest/'+str(category)+'/'):
            l = q.split('.')
            if(j>=1):
                break
            temp = cv2.imread('./temptest/'+str(category)+'/'+str(l[0])+'.jpg')
            res = cv2.matchTemplate(img,temp,cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if(max_val>0.7):
            
                img = cv2.circle(img,(int(max_loc[0]),int(max_loc[1])),1,(0,255,0),5)
                sp = temp.shape
                ma.append(max_val)
                j=j+1
                w = sp[0]
                h = sp[1]
                print(w)
                print(h)
                threshold = max(max_val,0.7)
                c = []
                d = []
                loc = np.where( res >= threshold)
                for pt in zip(*loc[::-1]):
                    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 3)
            
            
                #vis=img.copy()
                #cv2.imwrite('./test/' + str(l[0]) + '.jpg',vis)    
                cv2.imshow('result',img)
                cv2.waitKey(5)
                print(max_loc)
                print(str(l[0]))
                print(max_val)
                print('\n')
                cv2.waitKey(500)
                
        print(ma)
        print(len(ma))
        ma.sort()
        #print(ma[-1])

#otherfour('left_whirbone')
#otherfour('right_whirbone')
otherfour('left_corner')
otherfour('right_corner')
