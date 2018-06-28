#挑选labeled过的图，删除origin文件夹中未标注的图
import os
import cv2
import shutil

data = []
txtdir = "./Dataset0627/6/txt/"
oridir = "./Dataset0627/6/original_jpg/"
ntxtdir = "./Dataset0627/6/ntxt/"
noridir = "./Dataset0627/6/noriginal_jpg/"

for fname in os.listdir(r"./doc2/labeled_jpg"):
	#fname是选取的图片名
	s = fname[0:15]
	#s是txt文件名,phodir是原图文件名
	s += ".txt"
	datadir = txtdir + s
	phodir = oridir + fname
	shutil.copy(txtdir+s,ntxtdir+s)
	shutil.copy(oridir+fname,noridir+fname)
