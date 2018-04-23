clear
clc 
I=dicomread('/Users/liuchuanbin/Documents/MATLAB/CADDDH/dataset/dicomnew/5/5AMWR4A5/GAKHM1GB/I1000000');   %????????
% dcm4 = dicominfo('C:\\Users\\12264\\Desktop\\dicom new\\4\\5AMWR4A5\\GAKHM1GB\\I1000000');%????????
% dcm5 = dicominfo('C:\\Users\\12264\\Desktop\\dicom new\\5\\5AMWR4A5\\GAKHM1GB\\I1000000');%????????
% dcm6 = dicominfo('C:\\Users\\12264\\Desktop\\dicom new\\7\\5AMWR4A5\\GAKHM1GB\\I1000000');%????????
dcm5 = dicominfo('/Users/liuchuanbin/Documents/MATLAB/CADDDH/dataset/dicomnew/2/5AMWR4A5/GAKHM1GB/I1000000');
I = double(I);
I = I/4096*255;
I = uint8(I);
%%I = (I-dcm4.WindowCenter+0.5*dcm4.WindowWidth)/dcm4.WindowWidth*255;
imshow(I);%????????
hold on;
%%line([dcm5.CurveData_0(1) dcm5.CurveData_0(2)],[dcm5.CurveData_0(3) dcm5.CurveData_0(4)],'color','r','LineWidth',5);
for i = 1:1:6
    plot(dcm5.CurveData_0(i*2-1),dcm5.CurveData_0(i*2),'ro');
end
 for i = 1:1:4
     plot(dcm6.CurveData_2(i*2-1),dcm6.CurveData_2(i*2),'bo');
     plot(dcm6.CurveData_4(i*2-1),dcm6.CurveData_4(i*2),'bo');
 end
 plot(dcm6.CurveData_6(1),dcm6.CurveData_6(2),'yo');
 plot(dcm6.CurveData_8(1),dcm6.CurveData_8(2),'yo');