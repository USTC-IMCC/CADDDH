import os
import cv2
import shutil

f1 = open("label.txt",'w')
i = 0
for fname in os.listdir("./ntxt"):
    i += 1
    print(i)
    s = fname[0:15]
    f2 = open("./txt/"+fname,'r+')
    labels = f2.readlines()
    labels = labels[0]
    f1.write(s+'.jpg '+labels)
    print(s+'.jpg '+labels)
    f2.close()
f1.close()
