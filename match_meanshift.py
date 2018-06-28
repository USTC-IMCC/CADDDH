import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import numpy ,time
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np

for q in os.listdir(r'/home/hai/python_work/match_merge/voteresult/'):
    z=q.split('.')
    print (z[0])

    dataSet = []
    for m in range(1,81):
	
        img_rgb = cv2.imread('/home/hai/python_work/match_merge/voteresult/' + str(z[0]) + '.jpg')
        img_orinal = cv2.imread('/home/hai/python_work/match_merge/voteresult/' + str(z[0]) + '.jpg')
        #img_play = cv2.imread('/home/hai/python_work/match_merge/voteresult/' + str(z[0]) + '.jpg')
        j = 1
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        template = cv2.imread('/home/hai/python_work/match_merge/templeclub/' + str(m) + '.jpg',0)
        w, h = template.shape[::-1]
        print(w)
        print(h)
        res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.6
        a = []
        b = []
        loc = np.where( res >= threshold)
        for pt in zip(*loc[::-1]):
		
            if (((pt[0] + pt[0] + w)/2 < 800) and ((pt[1] + pt[1] + h)/2 < 800) and ((pt[0] + pt[0] + w)/2 > 200) and ((pt[1] + pt[1] + h)/2 > 200) ) :
                a.append((pt[0] + pt[0] + w)/2)
                b.append((pt[1] + pt[1] + h)/2)
            
                cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 1)
                print(a[j-1])
                print(b[j-1])
                print(j) 
                j = j + 1
        		
        cv2.imshow('result',img_rgb)
        print(sum(a)/j,sum(b)/j)
                
        if((sum(a)!=0) or (sum(b)!=0) ):

            imageo = cv2.circle(img_orinal,(int(sum(a)/j),int(sum(b)/j)),1,(255,0,255),3)
            viso = img_orinal.copy()
            cv2.imwrite('/home/hai/python_work/match_merge/voteresult/' + str(z[0]) + '.jpg',viso)

        #else:
	        #viso = img_orinal.copy()
	        #cv2.imwrite('/home/hai/python_work/match_template/voteinvalid/' + str(z[0]) + '.jpg',visu)
    
    
        cv2.waitKey(5)
		
        print ("step 1: load data...")
        
        if((sum(a)!=0) and (sum(b)!=0) ):
 
            dataSet.append([float(sum(a)/j), float(sum(b)/j)])
 
    numSamples = len(dataSet)
    if(numSamples < 5):
        vism = img_orinal.copy()
        cv2.imwrite('/home/hai/python_work/match_merge/voteresult/' + str(z[0]) + '.jpg',vism)
        continue
    X = np.array(dataSet)
 
    bandwidth = estimate_bandwidth(X, quantile=0.4)
    clf = MeanShift(bandwidth=bandwidth, bin_seeding=True,cluster_all=True).fit(X)
 
    centroids = clf.labels_
    print (centroids,type(centroids)) 
    
    arr_flag = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for i in clf.labels_:
        arr_flag[i]+=1
    k = 0
    for i in arr_flag:
        if(i > 3):
            k +=1
    print (k)
 
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    
    for i in range(numSamples):
        plt.plot(dataSet[i][0], dataSet[i][1], mark[clf.labels_[i]]) 
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    
    centroids =  clf.cluster_centers_
    for i in range(k):
        plt.plot(centroids[i][0], centroids[i][1], mark[i], markersize = 12)
    print (centroids) 
    print(centroids[0])

    imageo = cv2.circle(img_orinal,(int(centroids[0][0]),int(centroids[0][1])),1,(0,255,0),5)
    vism = img_orinal.copy()
    cv2.imwrite('/home/hai/python_work/match_merge/voteresult/' + str(z[0]) + '.jpg',vism)
    
    img_doctor = cv2.imread('/home/hai/python_work/match_merge/doctor/labeled/' + str(z[0]) + '.jpg')
    imaged = cv2.circle(img_doctor,(int(centroids[0][0]),int(centroids[0][1])),1,(0,255,0),5)
    visd = img_doctor.copy()
    cv2.imwrite('/home/hai/python_work/match_merge/doctor/labeled/' + str(z[0]) + '.jpg',visd)
    #plt.show()
        
    cv2.waitKey(5)
