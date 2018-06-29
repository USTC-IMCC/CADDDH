
import cv2

def point_show(filename):
    radius=10
    with open(filename) as file_project:
        content = file_project.readlines()
        for line in content:
            string = line.split()
            for i in range(len(string)):
                if string[i].endswith('.jpg'):
                    tmp_num = i
                    img = cv2.imread(string[i])
                    print img.shape
                    imgname = 'new'+string[i]
                    continue
                else:
                    if (i-tmp_num)%2 == 1:
                        x = int(round(float(string[i])))
                    else:
                        y = int(round(float(string[i])))
                        img[y,x,2]=255
                        cv2.circle(img,(x,y), radius, (0,0,255), -1)
                        cv2.imwrite(imgname,img)
        
        
filename = 'lfm/label.txt'
point_show(filename)
