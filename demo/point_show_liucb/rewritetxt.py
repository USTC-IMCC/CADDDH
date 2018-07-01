import os
import cv2
import shutil

f1 = open("predict.txt",'w')
i = 0
for fname in os.listdir("./txt_reg"):
    i += 1
    print(i)
    s = fname[0:15]
    f2 = open("./txt_reg/"+fname,'r+')
    labels = f2.readlines()
    labels = labels[0]
    f1.write(s+'.jpg '+labels)
    print(s+'.jpg '+labels)
    f2.close()
f1.close()
