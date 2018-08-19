# -*- coding:utf-8 -*-
import os
import struct
import pydicom
import argparse
import cv2
import shutil
import numpy as np

parser = argparse.ArgumentParser(description='Preprocee dicom data')
parser.add_argument('--dir',default='./dicom/',type=str,help='choose directory to process')
parser.add_argument('--res',default='./res/',type=str,help='choose directory to store the result')
parser.add_argument('--obj',default='stu',type=str,help='choose doc or stu to process')
args = parser.parse_args()
if args.dir[-1]!='/':	# in case user forget to print /
	args.dir += '/'
if args.res[-1]!='/':
	args.res += '/'

def transb2f(dcm)					:#transform binary to float
	data = []
	hexdata = b''
	try:
		for i in range(13):			#读取CurveData0~13
			a = dcm[0x5000+i*2,0x3000]	
			if i == 0:			#CurveData0有8个数据
				b = a[:32]
			else:
				b = a[:8]
			hexdata += b
	except IndexError:
		print('Unlabeled or Wrong Label!')
		return None
	except KeyError:
		print('Unlabeled or Wrong Label!')
		return None
	else:
		print('Successfully Labeled!')
	for i in range(32):				#标注了16个点共32个数据
		d = hexdata[4*i:4*i+4]			#一次读取2个字节为一个数据
		if (len(d) != 4):
			print('Too few labels!')
			return None
		f = struct.unpack('f',d)[0]
		data.append(f)
	return data

def finddicom(dcmdir):			#find the dir of dicom			
	for d1 in os.listdir(dcmdir):
		if(os.path.isdir(dcmdir+d1)):
			for d2 in os.listdir(dcmdir+d1):
				if(os.path.isdir(dcmdir+d1+'/'+d2)):
					for d3 in os.listdir(dcmdir+d1+'/'+d2):
						if(str(d3)=='I1000000'):
							return dcmdir+d1+'/'+d2+'/'+d3
	return None

def processdcm(dcmDIR,dcmID):	#save the data
	ori_path = args.res + 'original_jpg/'
	txt_path = args.res + 'txt/'
	dicom_path = args.res + 'dicom/'
	ljpg_path = args.res + 'labeled_jpg/'
	udicom_path = args.res + 'unlabeled_dicom/'
	cdata = ('0','1','2','3','4')
	ldata = ('l','L')
	rdata = ('r','R')
	if (dcmDIR != None):
		dcm = pydicom.read_file(dcmDIR)
		data = transb2f(dcm)
		if (data != None):		#经过标注的数据处理
			if (args.obj == 'doc'):
				if(len(dcmID) != 19 or (dcmID[15] not in ldata) or (dcmID[17] not in rdata) or (dcmID[16] not in cdata) or (dcmID[18] not in cdata)):#命名错误
					print('Name Erorr!')
					if not os.path.exists(udicom_path):
						os.makedirs(udicom_path)
					shutil.copyfile(dcmDIR,udicom_path+dcmID+'.dcm')
					return
			img = dcm.pixel_array
			if not os.path.exists(ori_path):
				os.makedirs(ori_path)
			img = (img - dcm.WindowCenter + 0.5 * dcm.WindowWidth)/dcm.WindowWidth * 255
			cv2.imwrite(ori_path + dcmID + '.jpg',img)
			if not os.path.exists(txt_path):
				os.makedirs(txt_path)
			s = ''
			with open(txt_path+dcmID+'.txt','w') as f:
				for i in range(32):
					s += str(data[i])
					s += ' '
				if(args.obj=='doc'):
					s += dcmID[16]+' '
					s += dcmID[18]+' '
				f.write(s)
			if not os.path.exists(dicom_path):
				os.makedirs(dicom_path)
			shutil.copyfile(dcmDIR,dicom_path+dcmID+'.dcm')
			if not os.path.exists(ljpg_path):
				os.makedirs(ljpg_path)
			limg = cv2.merge([img,img,img])
			for i in range(6):
				limg = drawcross(limg,data[2*i],data[2*i+1])
				cv2.putText(limg,str(i+1),(int(data[2*i])+15,int(data[2*i+1]-15)),cv2.FONT_HERSHEY_PLAIN,2,(255,100,0),3)
			cv2.imwrite(ljpg_path + dcmID + '.jpg',limg)
		else :
			if not os.path.exists(udicom_path):
				os.makedirs(udicom_path)
			shutil.copyfile(dcmDIR,udicom_path+dcmID+'.dcm')

def drawcross(img,ptx,pty,length = 10,width=2,color=(0,0,255)):
	p1 = int(ptx-length/2)
	p2 = int(pty-length/2)
	p3 = int(ptx+length/2)
	p4 = int(pty+length/2)
	p5 = int(ptx-length/2)
	p6 = int(pty+length/2)
	p7 = int(ptx+length/2)
	p8 = int(pty-length/2)
	img = cv2.line(img,(p1,p2),(p3,p4),color,width)
	img = cv2.line(img,(p5,p6),(p7,p8),color,width)
	return img


def main():
	for name in os.listdir(args.dir):
		if(os.path.isdir(args.dir+name)):			#标注者编号
			for name2 in os.listdir(args.dir+name):		#病人编号
				if(os.path.isdir(args.dir+name)):
					dcmDIR = args.dir+name+'/'+name2+'/'
					dcmID = name+'_'+name2
					dcmDIR = finddicom(dcmDIR)
					print(name2)
					processdcm(dcmDIR,dcmID)	

if __name__ == '__main__':
	main()
