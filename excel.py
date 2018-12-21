import os
import cv2
import math
import numpy as np
from PIL import Image, ImageFont, ImageDraw
llaz=0
lo=0
llao=0
llatw=0
llath=0
llaf=0

# doctor 0
ztllaz=0
ztlo=0
ztllao=0
ztllatw=0
ztllath=0
ztllaf=0
# doctor 1
otllaz=0
otlo=0
otllao=0
otllatw=0
otllath=0
otllaf=0
# doctor 2
twtllaz=0
twtlo=0
twtllao=0
twtllatw=0
twtllath=0
twtllaf=0
# doctor 3
thtllaz=0
thtlo=0
thtllao=0
thtllatw=0
thtllath=0
thtllaf=0
# doctor 4
ftllaz=0
ftlo=0
ftllao=0
ftllatw=0
ftllath=0
ftllaf=0

rlaz=0
ro=0
rlao=0
rlatw=0
rlath=0
rlaf=0

# doctor 0
ztrlaz=0
ztro=0
ztrlao=0
ztrlatw=0
ztrlath=0
ztrlaf=0
# doctor 1
otrlaz=0
otro=0
otrlao=0
otrlatw=0
otrlath=0
otrlaf=0
# doctor 2
twtrlaz=0
twtro=0
twtrlao=0
twtrlatw=0
twtrlath=0
twtrlaf=0
# doctor 3
thtrlaz=0
thtro=0
thtrlao=0
thtrlatw=0
thtrlath=0
thtrlaf=0
# doctor 4
ftrlaz=0
ftro=0
ftrlao=0
ftrlatw=0
ftrlath=0
ftrlaf=0
for q in os.listdir(r'./Dataset/pic/'):
    z=q.split('.')

    image = cv2.imread('./Dataset/pic/'+str(z[0])+'.jpg')
    sp = image.shape
    im = Image.open('./Dataset/pic/' + str(z[0]) + '.jpg')
    
    mal=0 #mark angle left
    mar=0 #mark angle right
    mpl=0 #mark place left
    mpr=0 #mark place right
    
    #print(sp[0],sp[1])
    data=list()
    with  open('./Dataset/txt/'+str(z[0])+'.txt','r') as fileReader:
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
			
        if(a1>22):
            if(x5>x3 and y5<y1):
                print('right whirbone place is:')
                print('I')
                mpr=1
		
            if(x5<x3 and y5<y1):
                print('right whirbone place is:')
                print('II')
                mpr=2
		
            if(x5<x3 and y5>y1 and y5<y3):
                print('right whirbone place is:')
                print('III')
                mpr=3
		
            if(x5<x3 and y5>y3):
                print('right whirbone place is:')
                print('IV')
                mpr=4
            
        #left whirbone place
        if(a2<22):
            mpl=0
			
        if(a2>22):
            if(x6<x4 and y6<y1):
                print('left whirbone place is:')
                print('I')
                mpl=1
		
            if(x6>x4 and y6<y1):
                print('left whirbone place is:')
                print('II')
                mpl=2
		
            if(x6>x4 and y6>y1 and y6<y4):
                print('left whirbone place is:')
                print('III')
                mpl=3
		
            if(x6>x4 and y6>y4):
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
		
        if(a1>22):
            if(A2>0):
                D1=A1*x5+B1*y5+C1
                #print(A1,B1,C1)
                D2=A2*x5+B2*y5+C2
                #print(A2,B2,C2)
                D3=A4*x5+B4*y5+C4
                #print(D1,D2,D3)
                if(D1>0 and D2>0):
                    #print(D1,D2)
                    print('right whirbone place is:')
                    print('I')
                    mpr=1
		
                if(D1>0 and D2<0):
                    #print(D1,D2)
                    print('right whirbone place is:')
                    print('II')
                    mpr=2
		
                if(D1<0 and D2<0 and D3<0):
                    #print(D1,D2,D3)
                    print('right whirbone place is:')
                    print('III')
                    mpr=3
		
                if(D1<0 and D2<0 and D3>0):
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
                if(D1>0 and D2<0):
                    #print(D1,D2)
                    print('right whirbone place is:')
                    print('I')
                    mpr=1
		
                if(D1>0 and D2>0):
                    #print(D1,D2)
                    print('right whirbone place is:')
                    print('II')
                    mpr=2
		
                if(D1<0 and D2>0 and D3<0):
                    #print(D1,D2,D3)
                    print('right whirbone place is:')
                    print('III')
                    mpr=3
		
                if(D1<0 and D2>0 and D3>0):
                    #print(D1,D2,D3)
                    print('right whirbone place is:')
                    print('IV')
                    mpr=4
            
            
            
        #left whirbone place
        if(a2<22):
            mpl=0
			
        if(a2>22):
            if(A2>0):
                D1=A1*x6+B1*y6+C1
                #print(A1,B1,C1)
                D4=A3*x6+B3*y6+C3
                #print(A3,B3,C3)
                D5=A5*x6+B5*y6+C5
                #print(D1,D4,D5)
                if(D1>0 and D4<0):
                    #print(D1,D4)
                    print('left whirbone place is:')
                    print('I')
                    mpl=1
		
                if(D1>0 and D4>0):
                    #print(D1,D4)
                    print('left whirbone place is:')
                    print('II')
                    mpl=2
		
                if(D1<0 and D4>0 and D5<0):
                    #print(D1,D4,D5)
                    print('left whirbone place is:')
                    print('III')
                    mpl=3
		
                if(D1<0 and D4>0 and D5>0):
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
                if(D1>0 and D4>0):
                    #print(D1,D4)
                    print('left whirbone place is:')
                    print('I')
                    mpl=1
		
                if(D1>0 and D4<0):
                    #print(D1,D4)
                    print('left whirbone place is:')
                    print('II')
                    mpl=2
		
                if(D1<0 and D4<0 and D5<0):
                    #print(D1,D4,D5)
                    print('left whirbone place is:')
                    print('III')
                    mpl=3
		
                if(D1<0 and D4<0 and D5>0):
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
    # left point
    if(str(z[0][16])==str(0)):
        llaz=llaz+1
        if(str(mpl)==str(0)):
            ztllaz=ztllaz+1
        if(str(mpl)==str(1)):
            ztllao=ztllao+1
        if(str(mpl)==str(2)):
            ztllatw=ztllatw+1
        if(str(mpl)==str(3)):
            ztllath=ztllath+1
        if(str(mpl)==str(4)):
            ztllaf=ztllaf+1	
		
    if(str(z[0][16])=='O'):
        lo=lo+1
        
			
    if(str(z[0][16])==str(1)):
        llao=llao+1	
        if(str(mpl)==str(0)):
            otllaz=otllaz+1
        if(str(mpl)==str(1)):
            otllao=otllao+1
        if(str(mpl)==str(2)):
            otllatw=otllatw+1
        if(str(mpl)==str(3)):
            otllath=otllath+1
        if(str(mpl)==str(4)):
            otllaf=otllaf+1	
			
    if(str(z[0][16])==str(2)):
        llatw=llatw+1
        if(str(mpl)==str(0)):
            twtllaz=twtllaz+1
        if(str(mpl)==str(1)):
            twtllao=twtllao+1
        if(str(mpl)==str(2)):
            twtllatw=twtllatw+1
        if(str(mpl)==str(3)):
            twtllath=twtllath+1
        if(str(mpl)==str(4)):
            twtllaf=twtllaf+1	
			        
    if(str(z[0][16])==str(3)):
        llath=llath+1
        if(str(mpl)==str(0)):
            thtllaz=thtllaz+1
        if(str(mpl)==str(1)):
            thtllao=thtllao+1
        if(str(mpl)==str(2)):
            thtllatw=thtllatw+1
        if(str(mpl)==str(3)):
            thtllath=thtllath+1
        if(str(mpl)==str(4)):
            thtllaf=thtllaf+1	
			
    if(str(z[0][16])==str(4)):
        llaf=llaf+1
        if(str(mpl)==str(0)):
            ftllaz=ftllaz+1
        if(str(mpl)==str(1)):
            ftllao=ftllao+1
        if(str(mpl)==str(2)):
            ftllatw=ftllatw+1
        if(str(mpl)==str(3)):
            ftllath=ftllath+1
        if(str(mpl)==str(4)):
            ftllaf=ftllaf+1	
        
    # right point    
    if(str(z[0][18])==str(0)):
        rlaz=rlaz+1
        if(str(mpr)==str(0)):
            ztrlaz=ztrlaz+1
        if(str(mpr)==str(1)):
            ztrlao=ztrlao+1
        if(str(mpr)==str(2)):
            ztrlatw=ztrlatw+1
        if(str(mpr)==str(3)):
            ztrlath=ztrlath+1
        if(str(mpr)==str(4)):
            ztrlaf=ztrlaf+1
			
    if(str(z[0][18])=='O'):
        ro=ro+1
        
			
    if(str(z[0][18])==str(1)):
        rlao=rlao+1	
        if(str(mpr)==str(0)):
            otrlaz=otrlaz+1
        if(str(mpr)==str(1)):
            otrlao=otrlao+1
        if(str(mpr)==str(2)):
            otrlatw=otrlatw+1
        if(str(mpr)==str(3)):
            otrlath=otrlath+1
        if(str(mpr)==str(4)):
           otrlaf=otrlaf+1
			
    if(str(z[0][18])==str(2)):
        rlatw=rlatw+1 
        if(str(mpr)==str(0)):
            twtrlaz=twtrlaz+1
        if(str(mpr)==str(1)):
            twtrlao=twtrlao+1
        if(str(mpr)==str(2)):
            twtrlatw=twtrlatw+1
        if(str(mpr)==str(3)):
            twtrlath=twtrlath+1
        if(str(mpr)==str(4)):
            twtrlaf=twtrlaf+1
			       
    if(str(z[0][18])==str(3)):
        rlath=rlath+1
        if(str(mpr)==str(0)):
            thtrlaz=thtrlaz+1
        if(str(mpr)==str(1)):
            thtrlao=thtrlao+1
        if(str(mpr)==str(2)):
            thtrlatw=thtrlatw+1
        if(str(mpr)==str(3)):
            thtrlath=thtrlath+1
        if(str(mpr)==str(4)):
            thtrlaf=thtrlaf+1
			
    if(str(z[0][18])==str(4)):
        rlaf=rlaf+1
        if(str(mpr)==str(0)):
            ftrlaz=ftrlaz+1
        if(str(mpr)==str(1)):
            ftrlao=ftrlao+1
        if(str(mpr)==str(2)):
            ftrlatw=ftrlatw+1
        if(str(mpr)==str(3)):
            ftrlath=ftrlath+1
        if(str(mpr)==str(4)):
            ftrlaf=ftrlaf+1
    
            
    print('\n')
with open('./compare.txt','a') as f:
    f.write('left marked 0 of doctor :' + str(llaz)) # doctor 0 left
    f.write('\n')
    f.write('left marked 0 of predict :' + str(ztllaz))
    f.write('\n')
    f.write('left marked 1 of predict :' + str(ztllao))
    f.write('\n')
    f.write('left marked 2 of predict :' + str(ztllatw))
    f.write('\n')
    f.write('left marked 3 of predict :' + str(ztllath))
    f.write('\n')
    f.write('left marked 4 of predict :' + str(ztllaf))
    f.write('\n')
    f.write('\n')
    
    f.write('left marked o of doctor :' + str(lo)) # doctor o left
    f.write('\n')
    f.write('\n')
    
    f.write('left marked 1 of doctor :' + str(llao)) # doctor 1 left
    f.write('\n')
    f.write('left marked 0 of predict :' + str(otllaz))
    f.write('\n')
    f.write('left marked 1 of predict :' + str(otllao))
    f.write('\n')
    f.write('left marked 2 of predict :' + str(otllatw))
    f.write('\n')
    f.write('left marked 3 of predict :' + str(otllath))
    f.write('\n')
    f.write('left marked 4 of predict :' + str(otllaf))
    f.write('\n')
    f.write('\n')
    
    f.write('left marked 2 of doctor :' + str(llatw)) # doctor 2 left
    f.write('\n')
    f.write('left marked 0 of predict :' + str(twtllaz))
    f.write('\n')
    f.write('left marked 1 of predict :' + str(twtllao))
    f.write('\n')
    f.write('left marked 2 of predict :' + str(twtllatw))
    f.write('\n')
    f.write('left marked 3 of predict :' + str(twtllath))
    f.write('\n')
    f.write('left marked 4 of predict :' + str(twtllaf))
    f.write('\n')
    f.write('\n')
    
    f.write('left marked 3 of doctor :' + str(llath)) # doctor 3 left
    f.write('\n')
    f.write('left marked 0 of predict :' + str(thtllaz))
    f.write('\n')
    f.write('left marked 1 of predict :' + str(thtllao))
    f.write('\n')
    f.write('left marked 2 of predict :' + str(thtllatw))
    f.write('\n')
    f.write('left marked 3 of predict :' + str(thtllath))
    f.write('\n')
    f.write('left marked 4 of predict :' + str(thtllaf))
    f.write('\n')
    f.write('\n')
    
    f.write('left marked 4 of doctor :' + str(llaf)) # doctor 4 left
    f.write('\n')
    f.write('left marked 0 of predict :' + str(ftllaz))
    f.write('\n')
    f.write('left marked 1 of predict :' + str(ftllao))
    f.write('\n')
    f.write('left marked 2 of predict :' + str(ftllatw))
    f.write('\n')
    f.write('left marked 3 of predict :' + str(ftllath))
    f.write('\n')
    f.write('left marked 4 of predict :' + str(ftllaf))
    f.write('\n')
    f.write('\n')
    f.write('\n')
    
    f.write('right marked 0 of doctor :' + str(rlaz)) #doctor 0 right
    f.write('\n')
    f.write('right marked 0 of predict :' + str(ztrlaz))
    f.write('\n')
    f.write('right marked 1 of predict :' + str(ztrlao))
    f.write('\n')
    f.write('right marked 2 of predict :' + str(ztrlatw))
    f.write('\n')
    f.write('right marked 3 of predict :' + str(ztrlath))
    f.write('\n')
    f.write('right marked 4 of predict :' + str(ztrlaf))
    f.write('\n')
    f.write('\n')
    
    f.write('right marked o of doctor :' + str(ro)) #doctor o right
    f.write('\n')
    f.write('\n')
    
    f.write('right marked 1 of doctor :' + str(rlao)) #doctor 1 right
    f.write('\n')
    f.write('right marked 0 of predict :' + str(otrlaz))
    f.write('\n')
    f.write('right marked 1 of predict :' + str(otrlao))
    f.write('\n')
    f.write('right marked 2 of predict :' + str(otrlatw))
    f.write('\n')
    f.write('right marked 3 of predict :' + str(otrlath))
    f.write('\n')
    f.write('right marked 4 of predict :' + str(otrlaf))
    f.write('\n')
    f.write('\n')
    
    f.write('right marked 2 of doctor :' + str(rlatw)) #doctor 2 right
    f.write('\n')
    f.write('right marked 0 of predict :' + str(twtrlaz))
    f.write('\n')
    f.write('right marked 1 of predict :' + str(twtrlao))
    f.write('\n')
    f.write('right marked 2 of predict :' + str(twtrlatw))
    f.write('\n')
    f.write('right marked 3 of predict :' + str(twtrlath))
    f.write('\n')
    f.write('right marked 4 of predict :' + str(twtrlaf))
    f.write('\n')
    f.write('\n')
    
    f.write('right marked 3 of doctor :' + str(rlath)) #doctor 3 right
    f.write('\n')
    f.write('right marked 0 of predict :' + str(thtrlaz))
    f.write('\n')
    f.write('right marked 1 of predict :' + str(thtrlao))
    f.write('\n')
    f.write('right marked 2 of predict :' + str(thtrlatw))
    f.write('\n')
    f.write('right marked 3 of predict :' + str(thtrlath))
    f.write('\n')
    f.write('right marked 4 of predict :' + str(thtrlaf))
    f.write('\n')
    f.write('\n')
    
    f.write('right marked 4 of doctor :' + str(rlaf)) #doctor 4 right
    f.write('\n')
    f.write('right marked 0 of predict :' + str(ftrlaz))
    f.write('\n')
    f.write('right marked 1 of predict :' + str(ftrlao))
    f.write('\n')
    f.write('right marked 2 of predict :' + str(ftrlatw))
    f.write('\n')
    f.write('right marked 3 of predict :' + str(ftrlath))
    f.write('\n')
    f.write('right marked 4 of predict :' + str(ftrlaf))
    f.write('\n')

