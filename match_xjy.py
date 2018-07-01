# -*- coding:utf-8 -*-
import cv2
import os
import numpy as np
from numpy import random,mat
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import numpy ,time
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
import PIL.Image as Image

oridir = './original_jpg/'
votdir = './voteresult/'
resdir = './res/'
labdir = './labeled_jpg/'
tempdir = "./temple/"

PN=(0,1,4,5)
sondir = []
sondir.append('left_center/')
sondir.append('right_center/')
sondir.append('left_coner/')
sondir.append('right_coner/')
sondir.append('left_whirbone/')
sondir.append('right_whirbone/')

#核心参数位置，六个参数对应六个点
thre  = (0.62,0.63,0.65,0.65,0.65,0.65,0.65)
scope = ((0.2,0.8,0.15,0.7),(0.2,0.8,0.3,0.85),(0.2,0.8,0.1,0.5),(0.2,0.8,0.5,0.9),(0.2,0.8,0.15,0.7),(0.2,0.8,0.3,0.85))#上下左右范围系数 
km    = (0.28,0.28,0.28,0.28,0.28,0.28)


def main():
	#P用来存储结果
	PData = []
	errorstr = 'Not Found'
	
	for q in os.listdir(r'./original_jpg'):
		z=q.split('.')
		print(oridir+q)
		img_orinal = cv2.imread(oridir+q)
		img_doctor = cv2.imread(labdir+q)
		#cv2.imwrite(resdir+q,img_doctor)
		#img_doc = cv3.imread(resdir+q)
		
	    #读取原图尺寸用于限制模板查找范围
		imgshape = img_orinal.shape
		wq = imgshape[1]
		hq = imgshape[0]
		PData = []
		for pn in PN:
			cv2.imwrite(votdir+sondir[pn]+q,img_orinal)
			img_vot = cv2.imread(votdir+sondir[pn]+q)
			dataSet = []
			#此处匹配所有模板并生成原始匹配数据
			dataSet = match(q,tempdir+sondir[pn],pn,img_orinal,img_vot,thre[pn],scope[pn])
			#计算数据kMeans重心
			(px,py) = kMeanCal(dataSet,km[pn],6)
			if(px==0 or py==0):
				cv2.putText(img_doctor, errorstr, (200,200), cv2.FONT_HERSHEY_DUPLEX, 2, (255,0, 255 ), 3); #蓝绿红
				visd = img_doctor.copy()
				cv2.imwrite(resdir+q,visd)
				continue
			imageo = cv2.circle(img_vot,(int(px),int(py)),1,(0,255,0),5)
			cv2.imwrite(votdir+sondir[pn]+q,imageo)
			PData.append((px,py))
			#print (px,py)
		print(PData)
		for pd in PData:
			imaged = cv2.circle(img_doctor,(int(pd[0]),int(pd[1])),1,(0,255,0),5)
			visd = imaged.copy()
			cv2.imwrite(resdir+q,visd)
		
		cv2.destroyAllWindows()
		#cv2.imshow('res',imaged)    
		#cv2.waitKey(500)


#tempdir是模板路径
#img_orinal是原图
#srange[4],0123对应上下左右的限度系数,如：上限系数×图像高度就是原图匹配模板的上线范围
def match(q,tempdir,n,img_orinal,img_vot,threshold=0.6,srange=(0.2,0.8,0.15,0.68)):
	data = []
	imgshape = img_orinal.shape
	wq = imgshape[1]
	hq = imgshape[0]
	for m in os.listdir(tempdir):		       
		cv2.imshow(q,img_vot)
		cv2.waitKey(5)
		j = 0
		img_gray = cv2.cvtColor(img_orinal, cv2.COLOR_BGR2GRAY)
		
		template = cv2.imread(tempdir+m,0)
		
		w, h = template.shape[::-1]
		#预处理模板（增强对比度）
		#img_gray = duibi(img_gray)
		#template = duibi(template)
		#print(w,h)
		res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
		#threshold = 0.6
		a = []
		#b = []
		loc = np.where( res >= threshold)
		jj=0
		for pt in zip(*loc[::-1]):		
			if (((pt[0] + pt[0] + w)/2 < srange[3]*wq) and ((pt[1] + pt[1] + h)/2 < srange[1]*hq) and ((pt[0] + pt[0] + w)/2 > srange[2]*wq) and ((pt[1] + pt[1] + h)/2 > srange[0]*hq) ) :
				if jj>50 and jj%3!=0:
					jj+=1
					continue
				if j>300:
					break
				a.append(((pt[0] + pt[0] + w)/2,((pt[1] + pt[1] + h)/2)))
				j = j + 1
				jj = jj+1
				
		#print('j=')
		#print (j)
		if(j>6):	
			(x,y) = kMeanCal(a,0.5,6)
			if(x==0 or y==0):
				continue
			else:
				imageo = cv2.circle(img_vot,(int(x),int(y)),1,(0,0,255),2)
				cv2.imwrite(votdir+sondir[n]+q,imageo)
				data.append((x,y))
	return data

#dataSet是输入数据
#N限制数据量
def kMeanCal(dataSet,k=0.3,N=6):
		numSamples = len(dataSet)
		#print(numSamples)
		if numSamples<N:
			return (0,0)
		X = np.array(dataSet)
	 
		bandwidth = estimate_bandwidth(X, k)
		if bandwidth==0:
			return(0,0)
		clf = MeanShift(bandwidth=bandwidth, bin_seeding=True,cluster_all=True).fit(X)
		centroids = clf.labels_
		#print (centroids,type(centroids)) 
	    
		arr_flag = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
		for i in clf.labels_:
			arr_flag[i]+=1
		k = 0
		for i in arr_flag:
			if(i > 3):
				k +=1
		#print (k)
	 
		mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr'] 
		mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
	    
		centroids =  clf.cluster_centers_
		for i in range(k):
			plt.plot(centroids[i][0], centroids[i][1], mark[i], markersize = 12)
		#print(centroids[0])
		return (centroids[0][0],centroids[0][1])

#增强图像对比度
def duibi(img,k1=1.6,k2=-160):
	img=cv2.imread('harris.jpg')
	#cv2.imshow('img',img)
	rows,cols,channels=img.shape
	dst=img.copy()

	for i in range(rows):
		for j in range(cols):
			for c in range(3):
				color=img[i,j][c]*k1 + k2
			if color>255:
				dst[i,j][c]=255
			elif color<0:
				dst[i,j][c]=0
			else:
				dst[i,j][c] = color
	return dst
	#cv2.imshow('dst',dst)
	
#cv2.waitKey(0)
#cv2.destroyAllWindows()

if __name__ == '__main__':
	main()







