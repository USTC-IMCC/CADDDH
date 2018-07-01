import cv2
import os

def point_doc_show(filename):
    radius=2
    with open(filename) as file_project:
        content = file_project.readlines()
        for line in content:
            string = line.split()
            for i in range(13):
                if string[i].endswith('.jpg'):
                    tmp_num = i
                    img = cv2.imread('./original_jpg/'+string[i])
                    print img.shape
                    imgname = './doc_label/'+string[i]
                    continue
                else:
                    if (i-tmp_num)%2 == 1:
                        x = int(round(float(string[i])))
                    else:
                        y = int(round(float(string[i])))
                        img[y,x,2]=255
                        cv2.circle(img,(x,y), radius, (0,255,0), -1)
                        cv2.imwrite(imgname,img)

def point_predict_show(filename):
    radius=2
    with open(filename) as file_project:
        content = file_project.readlines()
        for line in content:
            string = line.split()
            for i in range(13):
                if string[i].endswith('.jpg'):
                    tmp_num = i
                    img = cv2.imread('./doc_label/'+string[i])
                    print img.shape
                    imgname = './predict/'+string[i]
                    continue
                else:
                    if (i-tmp_num)%2 == 1:
                        x = int(round(float(string[i])))
                    else:
                        y = int(round(float(string[i])))
                        img[y,x,2]=255
                        cv2.circle(img,(x,y), radius, (0,0,255), -1)
                        cv2.imwrite(imgname,img)

if not os.path.isdir('doc_label'):
    os.mkdir('doc_label')
if not os.path.isdir('predict'):
    os.mkdir('predict')         
filename = './label.txt'
point_doc_show(filename)
filename = './predict.txt'
point_predict_show(filename)

