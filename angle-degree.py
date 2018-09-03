import os
import cv2
import math
import numpy as np
from PIL import Image, ImageFont, ImageDraw

for q in os.listdir(r'./doctest/pic/'):
    z=q.split('.')

    image = cv2.imread('./doctest/pic/'+str(z[0])+'.jpg')
    sp = image.shape
    im = Image.open('./doctest/pic/' + str(z[0]) + '.jpg')
    
    mal=0 #mark angle left
    mar=0 #mark angle right
    mpl=0 #mark place left
    mpr=0 #mark place right
    
    #print(sp[0],sp[1])
    data=list()
    with  open('./doctest/txt/'+str(z[0])+'.txt','r') as fileReader:
        lines = fileReader.readlines()#读取全部内容
        for line in lines:
            line = line.strip()
            line = line.split()#根据数据间的分隔符切割行数据
            data.append(line[:])
            #print(len(line))

    #for i in range(len(line)):
        #print(line[i])
        
    #1 point     
    x1=float(line[0])
    y1=sp[0]-float(line[1])
    
    #2 point     
    x2=float(line[2])
    y2=sp[0]-float(line[3])
    
    #3 point     
    x3=float(line[4])
    y3=sp[0]-float(line[5])
    
    #4 point     
    x4=float(line[6])
    y4=sp[0]-float(line[7])
    
    #5 point     
    x5=float(line[8])
    y5=sp[0]-float(line[9])
    
    #6 point     
    x6=float(line[10])
    y6=sp[0]-float(line[11])
    
    
    
    if(y1==y2): #level
		
		#right angle
        d1=abs(y3-y1)
        l1=math.sqrt((x3-x1)*(x3-x1)+(y3-y1)*(y3-y1))
        a1=math.degrees(math.asin(d1/l1))
        print(z[0])
        print('right angle: a1=')
        print(a1)
        mar=a1
        
        #left angle
        d2=abs(y4-y2)
        l2=math.sqrt((x4-x2)*(x4-x2)+(y4-y2)*(y4-y2))
        a2=math.degrees(math.asin(d2/l2))
        print(z[0])
        print('left angle: a2=')
        print(a2)
        mal=a2
        
        #right whirbone place
        if(a1<22):
            mpr=0
			
        if(a1>=22):
            if(x5>x3 and y5<y1):
                print('right whirbone place is:')
                print('I')
                mpr=1
		
            if(x5<=x3 and y5<y1):
                print('right whirbone place is:')
                print('II')
                mpr=2
		
            if(y5>=y1 and y5<y3):
			#if(x5<x3 and y5>=y1 and y5<y3):
                print('right whirbone place is:')
                print('III')
                mpr=3
		
            if(y5>=y3):
			#if(x5<x3 and y5>y3):	
                print('right whirbone place is:')
                print('IV')
                mpr=4
            
        #left whirbone place
        if(a2<22):
            mpl=0
			
        if(a2>=22):
            if(x6<x4 and y6<y1):
                print('left whirbone place is:')
                print('I')
                mpl=1
		
            if(x6>=x4 and y6<y1):
                print('left whirbone place is:')
                print('II')
                mpl=2
		
            if(y6>=y1 and y6<y4):
			#if(x6>x4 and y6>=y1 and y6<y4):
                print('left whirbone place is:')
                print('III')
                mpl=3
		
            if(y6>=y4):
			#if(x6>x4 and y6>y4):
                print('left whirbone place is:')
                print('IV')
                mpl=4
    
    
    if(y1!=y2): #oblique
        #line 1
        A1=y2-y1
        B1=x1-x2
        C1=x2*y1-x1*y2
    
        #line 2
        A2=B1/A1
        B2=-1
        C2=y3-(B1/A1)*x3
    
        #line 3
        A3=B1/A1
        B3=-1
        C3=y4-(B1/A1)*x4
    
        #line 4
        A4=A1/B1
        B4=1
        C4=-(y3+(A1/B1)*x3)
    
        #line 5
        A5=A1/B1
        B5=1
        C5=-(y4+(A1/B1)*x4)
    
        #right angle
        d1=abs((A1*x3+B1*y3+C1)/(math.sqrt(A1*A1+B1*B1)))
        l1=math.sqrt((x3-x1)*(x3-x1)+(y3-y1)*(y3-y1))
        a1=math.degrees(math.asin(d1/l1))
        print(z[0])
        print('right angle: a1=')
        print(a1)
        mar=a1
    
    
        #left angle
        d2=abs((A1*x4+B1*y4+C1)/(math.sqrt(A1*A1+B1*B1)))
        l2=math.sqrt((x4-x2)*(x4-x2)+(y4-y2)*(y4-y2))
        a2=math.degrees(math.asin(d2/l2))
        print(z[0])
        print('left angle: a2 =')
        print(a2)
        mal=a2
    
        #right whirbone place
        if(a1<22):
            mpr=0
		
        if(a1>=22):
            if(A2>0):
                D1=A1*x5+B1*y5+C1
                #print(A1,B1,C1)
                D2=A2*x5+B2*y5+C2
                #print(A2,B2,C2)
                D3=A4*x5+B4*y5+C4
                #print(D1,D2,D3)
                if(D1>=0 and D2>0):
                    #print(D1,D2)
                    print('right whirbone place is:')
                    print('I')
                    mpr=1
		
                if(D1>0 and D2<=0):
                    #print(D1,D2)
                    print('right whirbone place is:')
                    print('II')
                    mpr=2
		
                if(D1<=0 and D3<0):
				#if(D1<=0 and D2<0 and D3<0):
                    #print(D1,D2,D3)
                    print('right whirbone place is:')
                    print('III')
                    mpr=3
		
                if(D1<0 and D3>=0):
				#if(D1<0 and D2<0 and D3>0):
                    #print(D1,D2,D3)
                    print('right whirbone place is:')
                    print('IV')
                    mpr=4
            
            if(A2<0):
                D1=A1*x5+B1*y5+C1
                #print(A1,B1,C1)
                D2=A2*x5+B2*y5+C2
                #print(A2,B2,C2)
                D3=A4*x5+B4*y5+C4
                #print(D1,D2,D3)
                if(D1>=0 and D2<0):
                    #print(D1,D2)
                    print('right whirbone place is:')
                    print('I')
                    mpr=1
		
                if(D1>0 and D2>=0):
                    #print(D1,D2)
                    print('right whirbone place is:')
                    print('II')
                    mpr=2
		
                if(D1<=0 and D3<0):
				#if(D1<=0 and D2>0 and D3<0):
                    #print(D1,D2,D3)
                    print('right whirbone place is:')
                    print('III')
                    mpr=3
		
                if(D1<0 and D3>=0):
				#if(D1<0 and D2>0 and D3>0):
                    #print(D1,D2,D3)
                    print('right whirbone place is:')
                    print('IV')
                    mpr=4
            
            
            
        #left whirbone place
        if(a2<22):
            mpl=0
			
        if(a2>=22):
            if(A2>0):
                D1=A1*x6+B1*y6+C1
                #print(A1,B1,C1)
                D4=A3*x6+B3*y6+C3
                #print(A3,B3,C3)
                D5=A5*x6+B5*y6+C5
                #print(D1,D4,D5)
                if(D1>=0 and D4<0):
                    #print(D1,D4)
                    print('left whirbone place is:')
                    print('I')
                    mpl=1
		
                if(D1>0 and D4>=0):
                    #print(D1,D4)
                    print('left whirbone place is:')
                    print('II')
                    mpl=2
		
                if(D1<=0 and D5<0):
				#if(D1<=0 and D4>0 and D5<0):
                    #print(D1,D4,D5)
                    print('left whirbone place is:')
                    print('III')
                    mpl=3
		
                if(D1<0 and D5>=0):
				#if(D1<0 and D4>0 and D5>0):
                    #print(D1,D4,D5)
                    print('left whirbone place is:')
                    print('IV')
                    mpl=4
            
            if(A2<0):
                D1=A1*x6+B1*y6+C1
                #print(A1,B1,C1)
                D4=A3*x6+B3*y6+C3
                #print(A3,B3,C3)
                D5=A5*x6+B5*y6+C5
                #print(D1,D4,D5)
                if(D1>=0 and D4>0):
                    #print(D1,D4)
                    print('left whirbone place is:')
                    print('I')
                    mpl=1
		
                if(D1>0 and D4<=0):
                    #print(D1,D4)
                    print('left whirbone place is:')
                    print('II')
                    mpl=2
		
                if(D1<=0 and D5<0):
				#if(D1<=0 and D4<0 and D5<0):
                    #print(D1,D4,D5)
                    print('left whirbone place is:')
                    print('III')
                    mpl=3
		
                if(D1<0 and D5>=0):
				#if(D1<0 and D4<0 and D5>0):
                    #print(D1,D4,D5)
                    print('left whirbone place is:')
                    print('IV')
                    mpl=4
            
    print(mar,mal,mpr,mpl)
    draw = ImageDraw.Draw(im)
    ft = ImageFont.truetype('/usr/share/fonts/truetype/tlwg/Garuda.ttf', 50)
    draw.text((150,20), str('Right angle = ' + str(mar)), font = ft, fill = 'red')
    draw.text((150,70), str('Left angle = ' + str(mal)), font = ft, fill = 'red')
    draw.text((150,170), str('Right place = ' + str(mpr)), font = ft, fill = 'green')
    draw.text((150,220), str('Left place = ' + str(mpl)), font = ft, fill = 'green')
    im.save('./compare/' + str(z[0]) + '.jpg')
    with open('./compare.txt','a') as f:
		
        if(str(z[0][18])==str(mpr)):
            R='ok'
        else:
            R='er'
        if(str(z[0][16])==str(mpl)):
            L='ok'
        else:
            L='er'
			
        f.write('Picture name: '+str(z[0])+'   Doc Right: '+str(z[0][18])+'   Doc Left: '+str(z[0][16])+'   Test Right: '+str(mpr)+'   Test Left: '+str(mpl)+'   R: '+str(R)+'   L: '+str(L))
        f.write('\n')

    print('\n')

