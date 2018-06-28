import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import numpy ,time
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np


#for q in range(1,21):


for q in os.listdir(r'/home/hai/python_work/match_merge/voteresult/'):
    z=q.split('.')
    print (z[0])

    dataSet = []
    for m in range(1,39):
	
	
        #for t in range(q,q+1):
	
        img_rgb = cv2.imread('/home/hai/python_work/match_merge/voteresult/' + str(z[0]) + '.jpg')
        img_orinal = cv2.imread('/home/hai/python_work/match_merge/voteresult/' + str(z[0]) + '.jpg')
        #img_play = cv2.imread('/home/hai/python_work/match_merge/voteresult/' + str(z[0]) + '.jpg')
        j = 1
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        template = cv2.imread('/home/hai/python_work/match_merge/templeclub/' + str(m) + '.png',0)
        w, h = template.shape[::-1]
        print(w)
        print(h)
        res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.6
        a = []
        b = []
        loc = np.where( res >= threshold)
        for pt in zip(*loc[::-1]):
		
          
       
            if (((pt[0] + pt[0] + w)/2 < 600) and ((pt[1] + pt[1] + h)/2 < 700) and ((pt[0] + pt[0] + w)/2 > 200) and ((pt[1] + pt[1] + h)/2 > 200) ) :
                a.append((pt[0] + pt[0] + w)/2)
                b.append((pt[1] + pt[1] + h)/2)
            
        
                cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 1)
                print(a[j-1])
                print(b[j-1])
                print(j) 
                j = j + 1
        
				
        cv2.imshow('result',img_rgb)
        print(sum(a)/j,sum(b)/j)
            
            
            
        #if((sum(a)!=0) and (sum(b)!=0) ):
    
    
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
    if(numSamples < 9):
        vism = img_orinal.copy()
        cv2.imwrite('/home/hai/python_work/match_merge/voteresult/' + str(z[0]) + '.jpg',vism)
        continue
    X = np.array(dataSet) #列表类型转换成array数组类型
 
    bandwidth = estimate_bandwidth(X, quantile=0.3, n_samples=500)
    clf = MeanShift(bandwidth=bandwidth, bin_seeding=True,cluster_all=True).fit(X)
 
    centroids = clf.labels_
    print (centroids,type(centroids)) #显示每一个点的聚类归属
    # 计算其自动生成的k，并将聚类数量小于3的排除
    arr_flag = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    for i in clf.labels_:
        arr_flag[i]+=1
    k = 0
    for i in arr_flag:
        if(i > 3):
            k +=1
    print (k)
 
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    #画出所有样例点 属于同一分类的绘制同样的颜色
    for i in range(numSamples):
        plt.plot(dataSet[i][0], dataSet[i][1], mark[clf.labels_[i]]) #mark[markIndex])
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # 画出质点，用特殊图型
    centroids =  clf.cluster_centers_
    for i in range(k):
        plt.plot(centroids[i][0], centroids[i][1], mark[i], markersize = 12)
    print (centroids) #显示中心点坐标
    print(centroids[0])


    imageo = cv2.circle(img_orinal,(int(centroids[0][0]),int(centroids[0][1])),1,(0,255,0),5)
    vism = img_orinal.copy()
    cv2.imwrite('/home/hai/python_work/match_merge/voteresult/' + str(z[0]) + '.jpg',vism)
    #plt.show()
        
    cv2.waitKey(5)

