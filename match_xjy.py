# -*- coding:utf-8 -*-
"""
Created on Thu July 2 2018
@author: xujingyuan
"""
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
labdir = './txt_doc/'
tempdir = "./temple/"
txtdir = './txt/'

search_area = (0.75,0.85,0.15,0.85)#上下左右
ratio = 0.1#搜索区域前10%亮度的点作为阈值
resizeN=8
PN=(0,1,2,3,4,5)
ori_size = 280

sondir = []
sondir.append('left_center/')
sondir.append('right_center/')
sondir.append('left_coner/')
sondir.append('right_coner/')
sondir.append('left_whirbone/')
sondir.append('right_whirbone/')

#核心参数位置，六个参数对应六个点
thre  = (0.62,0.63,0.65,0.65,0.65,0.65,0.65)
#scope = ((0.23,0.78,0.15,0.7),(0.2,0.8,0.3,0.85),(0.2,0.8,0.1,0.5),(0.2,0.8,0.5,0.9),(0.2,0.8,0.15,0.7),(0.2,0.8,0.3,0.85))#上下左右范围系数 
km    = (0.28,0.28,0.28,0.28,0.28,0.28)


def main():
	#P用来存储结果
	PData = []
	errorstr = 'Not Found'
	
	for q in os.listdir(r'./original_jpg'):
		print(oridir+q)
		img_ori = cv2.imread(oridir+q)
		img_doctor = img_ori.copy()
		#读取测试集数据并标点显示在结果中作为对比
		z=q.split('.')
		stxt = z[0]+'.txt'
		ftxt = open(labdir+stxt,'r')
		linedoc = ftxt.readline()
		docdata = linedoc.split(' ')
		for idoc in range(0,6):
			doc1 = int(float(docdata[2*idoc]))
			doc2 = int(float(docdata[2*idoc+1]))
			imaged = cv2.circle(img_doctor,(doc1,doc2),1,(255,0,0),6)
		#确定搜索区域
		spand = findedge(img_ori)
		if img_ori.shape[1]/spand[2]>5:
			#无法信任初步findege的框选，使用默认框选
			spand = (int(img_ori.shape[1]*0.2),int(img_ori.shape[0]*0.2),int(img_ori.shape[1]*0.6),int(img_ori.shape[0]*0.6))
		img_orinal = img_ori[spand[1]:spand[1]+spand[3],spand[0]:spand[0]+spand[2]]
		#cv2.imshow('searcharea',img_orinal)
		#cv2.waitKey(0)
	    #搜索范围的尺寸
		imgshape = img_orinal.shape
		wq = imgshape[1]
		hq = imgshape[0]
		PData = []
		for pn in PN:
			#cv2.imshow('searcharea',img_ori)
			#cv2.waitKey(0)
			cv2.imwrite(votdir+sondir[pn]+q,img_ori)
			img_vot = cv2.imread(votdir+sondir[pn]+q)
			print(img_vot.shape)
			dataSet = []
			#此处匹配所有模板并生成原始匹配数据
			if pn==0 or pn==1:
				dataSet = match(spand,q,tempdir+sondir[pn],pn,img_orinal,img_vot,thre[pn])
			elif pn==2 or pn==4:
				#按照pn=0的点相对位置寻找该点
				top = int(PData[0][1]-ori_size/2)
				bottom = int(PData[0][1]+ori_size/2)
				left = int(PData[0][0]-ori_size)
				right = int(PData[0][0])
				img_left = img_ori[top:bottom,left:right]
				#cv2.imshow('left',img_left)
				spand2 = (left,top,ori_size,ori_size)
				#cv2.waitKey(0)
				dataSet = match(spand2,q,tempdir+sondir[pn],pn,img_left,img_vot,thre[pn])
			elif pn==3 or pn==5:
				#按照pn=0的点相对位置寻找该点
				top = int(PData[1][1]-ori_size/2)
				bottom = int(PData[1][1]+ori_size/2)
				left = int(PData[1][0])
				right = int(PData[1][0]+ori_size)
				img_right = img_ori[top:bottom,left:right]
				#cv2.imshow('right',img_right)
				spand2 = (left,top,ori_size,ori_size)
				#cv2.waitKey(0)
				dataSet = match(spand2,q,tempdir+sondir[pn],pn,img_right,img_vot,thre[pn])
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
			imaged = cv2.circle(img_doctor,(int(pd[0]),int(pd[1])),1,(0,255,0),4)
			visd = imaged.copy()
			cv2.imwrite(resdir+q,visd)
		#将结果写入txt文件
		
		z = q.split('.')
		#print(z[0])
		s = ''
		with open(txtdir+z[0]+'.txt','w') as f:
			for ppd in PData:
				s += str(ppd[0])+' '
				s += str(ppd[1])+' '
			f.write(s)
		
		
		cv2.destroyAllWindows()
		#cv2.imshow('res',imaged)    
		#cv2.waitKey(500)


#tempdir是模板路径
#img_orinal是原图
#srange[4],0123对应上下左右的限度系数,如：上限系数×图像高度就是原图匹配模板的上线范围
def match(spand,q,tempdir,n,img_orinal,img_vot,threshold=0.6,method=0):
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
		#预处理模板（做放缩）
		if n == 0 or n == 1:
			proposal_size = wq / resizeN
			rr = float(w)/proposal_size
			#print (rr)
			template = resize(template,rr)
		w1, h1 = template.shape[::-1]
		#print (img_gray.shape)
		#print (template.shape)
		res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
		#threshold = 0.6
		a = []
		#b = []	
		if method == 0:
			jj=0
			loc = np.where( res >= threshold)
			for pt in zip(*loc[::-1]):		
				if jj>50 and jj%3!=0:
					jj+=1
					continue
				if j>300:
					break
				a.append(((pt[0] + pt[0] + w1)/2,((pt[1] + pt[1] + h1)/2)))
				j = j + 1
				jj = jj+1
			if(j>6):	
				(x,y) = kMeanCal(a,0.5,6)
				if(x==0 or y==0):
					continue
				else:
					x+=spand[0]
					y+=spand[1]
					imageo = cv2.circle(img_vot,(int(x),int(y)),1,(0,0,255),2)
					cv2.imwrite(votdir+sondir[n]+q,imageo)
					
					data.append((x,y))
		elif method == 1:
			min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
			print(cv2.minMaxLoc(res))
			(x,y) = (max_loc[0],max_loc[1])
			pt = (x,y)
			if (((pt[0] + pt[0] + w)/2 < srange[3]*wq) and ((pt[1] + pt[1] + h)/2 < srange[1]*hq) and ((pt[0] + pt[0] + w)/2 > srange[2]*wq) and ((pt[1] + pt[1] + h)/2 > srange[0]*hq) ) :
				data.append((x,y))
				imageo = cv2.circle(img_vot,(int(x),int(y)),1,(0,0,255),2)
				cv2.imwrite(votdir+sondir[n]+q,imageo)
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

#寻找图像边缘
def findedge(img,r=ratio,area = search_area):
	w = img.shape[1]
	h = img.shape[0]
	#img1 = img[int(h*area[0]):int(h*area[1]),int(w*area[2]):int(w*area[3])]
	img1 = img.copy()
	gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	#cv2.imshow('1',gray)
	#cv2.waitKey(0)
	gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
	gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
	gradient = cv2.subtract(gradX, gradY)
	gradient = cv2.convertScaleAbs(gradient)
	#cv2.imshow('2',gradient)
	#cv2.waitKey(0)
	blurred = cv2.blur(gradient, (6, 6))
	(_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
	#cv2.imshow('3',thresh)
	#cv2.waitKey(0)
	
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
	closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
	
	closed = cv2.erode(closed, None, iterations=8)
	closed = cv2.dilate(closed, None, iterations=8)
	closed = cv2.dilate(closed, None, iterations=6)
	closed = cv2.erode(closed, None, iterations=6)
	
	for pz in range(100):
		for px in range(h):
			closed[px,pz] = 0
			closed[px,w-1-pz] = 0
		for py in range(w):
			closed[pz,py] = 0
			closed[h-pz-1,py] = 0
	#cv2.imshow('5',closed)
	#cv2.waitKey(0)
	(__,cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	maxscore=0
	
	for cc in range(len(cnts)):
		xx,yy ,ww,hh = cv2.boundingRect(cnts[cc])
		if ww + hh>maxscore:
			maxscore = ww + hh
			mx,my ,mw,mh = xx,yy,ww,hh
	#cv2.rectangle(img,(mx,my),(mx+mw,my+mh),(0,255,0),2)
	#cv2.imshow("Image", img)
	#cv2.waitKey(0)
	pt = (mx,my,mw,mh)
	return pt

#增强图像对比度
def duibi(img,k1=1.6,k2=-160):
	#img=cv2.imread('harris.jpg')
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
	
def resize(img,prop):
	w = img.shape[1]
	h = img.shape[0]
	reimg = cv2.resize(img,(int(w/prop),int(h/prop)),interpolation=cv2.INTER_CUBIC)
	return reimg

if __name__ == '__main__':
	main()







