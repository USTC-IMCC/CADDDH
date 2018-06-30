import os
import cv2
import shutil

data = []
txtdir = "./txt/"
oridir = "./original_jpg/"
ntxtdir = "./ntxt/"
noridir = "./noriginal_jpg/"

for fname in os.listdir("./labeled_jpg"):
	s = fname[0:15]
	s += ".txt"
	datadir = txtdir + s
	phodir = oridir + fname
	shutil.copy(txtdir+s,ntxtdir+s)
	shutil.copy(oridir+fname,noridir+fname)
