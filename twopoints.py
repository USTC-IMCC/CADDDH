import cv2
import os
import numpy as np

def twopoint(category,n,x1,x2,y1,y2,kuan,h):
    
    img = cv2.imread('./testdoctor/'+str(category)+'/' + str(n[0]) + '.jpg')
    sq = img.shape    
    ma = []
    j=0
    for q in os.listdir(r'./temptest/'+str(category)+'/'):
        l = q.split('.')
        if(j>=1):
            break
        temp = cv2.imread('./temptest/'+str(category)+'/'+str(l[0])+'.jpg')
        res = cv2.matchTemplate(img,temp,cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if(max_val>0.6):
            
            img = cv2.circle(img,(int(max_loc[0]),int(max_loc[1])),1,(0,255,0),5)
            sp = temp.shape
            ma.append(max_val)
            w = sp[0]
            h = sp[1]
            print(w)
            print(h)
            threshold = max(max_val,0.6)
            c = []
            d = []
            loc = np.where( res >= threshold)
            for pt in zip(*loc[::-1]):
                print(x1)
                print(x2)
                print(sp[0])
                print(sp[1])
                if (((pt[0]+pt[0]+w)/2 < x2) and ((pt[1]+pt[1]+h)/2 < 0.7*sq[0]) and ((pt[0]+pt[0]+w)/2 > x1) and ((pt[1]+pt[1]+h)/2 > 0.3*sq[0])) :
                
                    cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,0,255), 3)
                    
                    imaged = cv2.circle(img,(int((pt[0]+pt[0]+w)/2),int((pt[1]+pt[1]+h)/2)),1,(0,255,255),5)
                    visd = imaged.copy()
                    cv2.imwrite('./result/' + str(n[0]) + '.jpg',visd)
                    j=j+1
            
            
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




category='right_center'
for y in os.listdir(r'./testdoctor/'+str(category)+'/'):
    n = y.split('.')
    img = cv2.imread('./testdoctor/'+str(category)+'/' + str(n[0]) + '.jpg')
    sp = img.shape
    cropimg = img[100:sp[0]-100,50:sp[1]-50]
    ret, thresh = cv2.threshold(cv2.cvtColor(cropimg.copy(), cv2.COLOR_BGR2GRAY), 80, 255, cv2.THRESH_BINARY)
    image, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    kk = 0
    hh = 0
    c1 = 0
    c2 = 0
    d1 = 0
    d2 = 0
    e1 = 0
    e2 = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c) 
        if((w > 400) and (h > 400)):
            cv2.rectangle(cropimg, (x,y), (x+w, y+h), (0, 0, 0), 2)
            print(str(n[0]))
            print('\n')
            print(w)
            kk = w
            print(h)
            hh = h
            print(x)
            print(y)
            c1 = x+w/2+50
            d1 = c1-(w/2)
            d2 = c1+(w/2)
            c2 = y+h/2+100
            e1 = c2-(h/2)
            e2 = c2+(h/2)
            print(c1)
            print(d1)
            print(d2)
            print(c2)
            cv2.waitKey(5)
            #twopoint(category='left_center',n=n,x1=(d1+10),x2=(c1-10),y1=e1,y2=e2,kuan=kk,h=hh)
            twopoint(category='right_center',n=n,x1=(c1+10),x2=(d2-10),y1=e1,y2=e2,kuan=kk,h=hh)




