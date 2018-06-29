# -*- coding:utf-8 -*-
#遍历PIC文件夹，以此每幅图生成六个模板
import os
import cv2

size=300

data = []
txtdir = "./Dataset0628/txt/"
oridir = "./Dataset0628/original_jpg/"
#labdir = "./Dataset0628/6/laleled_jpg/"
wdir = "./res/"
tdir = "./temple/"

for fname in os.listdir(r"./PIC"):
	#fname是选取的图片名
	s = fname[0:15]
	#s是文件名
	s += ".txt"
	datadir = txtdir + s
	phodir = oridir + fname
	#lphodir = labdir + fname

	f = open(datadir)
	#line读取了所有数据，需要存取到一个列表内
	line = f.readline()
	#print(line)
	#按照空格分割字符串得到坐标信息
	temp = line.split(' ')
	img = cv2.imread(phodir)
	for i in range(0,6):
		d1 = int(float(temp[2*i]))
		d2 = int(float(temp[2*i+1]))
		#print(d1,d2)
		#切割图片并存储相应路径
		if i==0:
			timg = img[d2-size:d2+size,d1-size:d1+size]
			cv2.imwrite(tdir+"left_center/"+fname,timg)
		elif i==1:
			timg = img[d2-size:d2+size,d1-size:d1+size]
			cv2.imwrite(tdir+"right_center/"+fname,timg)
		elif i==2:
			timg = img[d2-size:d2+size,d1-size:d1+size]
			cv2.imwrite(tdir+"left_coner/"+fname,timg)
		elif i==3:
			timg = img[d2-size:d2+size,d1-size:d1+size]
			cv2.imwrite(tdir+"right_coner/"+fname,timg)
		elif i==4:
			timg = img[d2-size:d2+size,d1-size:d1+size]
			cv2.imwrite(tdir+"left_whirbone/"+fname,timg)
		elif i==5:
			timg = img[d2-size:d2+size,d1-size:d1+size]
			cv2.imwrite(tdir+"right_whirbone/"+fname,timg)
		#imageo = cv2.circle(img,(d1,d2),1,(255,0,255),3)

	#cv2.imwrite(wdir+fname,imageo)
	#cv2.imshow('result',imageo)
	
	print(fname)
	#data = []
	f.close
	#cv2.waitKey(2000)

