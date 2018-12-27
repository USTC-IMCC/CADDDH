import os
import cv2
import math
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import ker
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
mal=0 #mark angle left
mar=0 #mark angle right
mpl=0 #mark place left
mpr=0 #mark place right
for q in os.listdir(r'./Dataset/pic/'):
    z=q.split('.')
    
    mar,mal,mpr,mpl = ker.compute(z[0])

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

