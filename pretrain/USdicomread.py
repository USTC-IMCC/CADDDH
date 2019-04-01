# -*- coding: utf-8 -*-
'''
Created on Nov 26,2018
@author xjy
'''

import pydicom
import os
import numpy
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='Preprocee dicom data')
parser.add_argument('--dir',default='190329',type=str,help='choose directory to process')
args = parser.parse_args()
srcdir = '/home/liucb/data/ddh/us/'+args.dir+'/'

def main():
    for name in os.listdir(srcdir):
        #print(name)
        try:
            dcm = pydicom.read_file(srcdir+name)
            slices = dcm.pixel_array
            print(slices.shape,slices.ndim)#there are 2,3,4 dimention of slices
            dstdir = './'+args.dir+'/'+name#dstdir indicates destination
            if not os.path.exists(dstdir):
                os.makedirs(dstdir)
            if(slices.ndim==2):
                img = Image.fromarray(slices)
                img.save(dstdir+'/0.jpg')
                print('Saved!')
            if(slices.ndim==3):
                img = Image.fromarray(slices,'RGB')
                img.save(dstdir+'/0.jpg')
            if(slices.ndim==4):
                N = slices.shape[0]
                for i in range(N):
                    img = slices[i,:,:,:,]
                    img = Image.fromarray(img, 'RGB')
                    img.save(dstdir+'/'+str(i)+'.jpg')
        except Exception as e:
            print(name+str(e))

if __name__ == '__main__':
    main()

'''
dcm = pydicom.read_file("IM_0001")
img = dcm.pixel_array
img2 = img[0,:,:,:,]
print(type(img2))
img2 = Image.fromarray(img2, 'RGB')
img2.save('test.jpg')'''
