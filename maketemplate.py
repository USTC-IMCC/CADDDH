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

tempsize = 120
srcdir = './original_jpg/'
datadir = './txt_doc/'
tempdir = './template/'
ldir = './left_corner/'

def mkdir(path):
	path = path.strip()
	isExist = os.path.exists(path)
	if not isExist:
		os.makedirs(path)
		print (path,'Successfully Makedir...')
	else:
		print(path,'Already Exist..')

def matchscore(img,temp):
	res = cv2.matchTemplate(img,temp,cv2.TM_CCOEFF_NORMED)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	print(max_val)
	return max_val
	
def main():
	loop=0#循环读图数量
	j = 0#划分模板种类
	tt = []
	for q in os.listdir(srcdir):
		img_src = cv2.imread(srcdir+q)
		print (q)
		z = q.split('.')
		#读取模板数据
		f = open(datadir+z[0]+'.txt')
		line = f.readline()
		data = line.split(' ')
		x,y = int(float(data[0]))-tempsize/2,int(float(data[1]))-tempsize/2
		img = img_src[y:y+tempsize,x:x+tempsize]
		if loop==0:#第一张图直接处理
			tt.append(q)
			kinddir = tempdir+str(j)+'/'
			mkdir(kinddir)
			cv2.imwrite(kinddir+q,img)
			cv2.imwrite(ldir+str(j)+'.jpg',img)
			j+=1
		else:
			for i in range(j):
				tmp = cv2.imread(tempdir+str(i)+'/'+tt[i])
				if matchscore(img,tmp)>0.58:
					cv2.imwrite(tempdir+str(i)+'/'+q,tmp)
					break
				if i==j-1:#到了最后一个仍然没能匹配
					tt.append(q)
					kinddir = tempdir + str(j)+'/'
					mkdir(kinddir)
					cv2.imwrite(kinddir+q,img)
					cv2.imwrite(ldir+str(j)+'.jpg',img)
					j+=1
		#if loop>2000 or j==64:
		#	break
		loop+=1
	print (loop)
if __name__ == '__main__':
	main()
