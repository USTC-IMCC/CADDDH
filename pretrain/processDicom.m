function [ result ] = processDicom( path,labeler,window_width,window_center )
%function [ result ] = processDicom( path,labeler,dicom_path,txt_path,jpg1_path,jpg2_path,window_width,window_center )
%   此处显示详细说明
    try
        img = dicomread(path);   %读取图像
        dcm = dicominfo(path);%存储信息
        if (size(img,1) == 0)||(size(img,2) == 0)
            result = -4;
            return;
        end
        try
            try_window = dcm.WindowCenter;
        catch
            result = -4;
            return;
        end
    catch
        result = -4;
        return;
    end
    cur_path = 'H:\';
    dicom_path = [cur_path,'dicom\'];
    txt_path = [cur_path,'txt\'];
    jpg1_path = [cur_path,'original_jpg\'];
    jpg2_path = [cur_path,'labeled_jpg\'];
    unlabed_path = [cur_path,'unlabeddicom\'];
    path_list = {dicom_path,txt_path,jpg1_path,jpg2_path,unlabed_path};
    for i = 1:1:5
        if ~exist(path_list{i})
            mkdir(path_list{i});
        end
    end
    if(nargin<4)
        if(nargin<2)
            result = -1;
            return;
        end
        window_width = dcm.WindowWidth;
        window_center = dcm.WindowCenter;
    end
    if ~(strcmp(dicom_path(end),'\')||strcmp(dicom_path(end),'\'))
        dicom_path = [dicom_path,'\\'];
    end
    
    uid = dcm.AccessionNumber;
    try
        data1 = dcm.CurveData_0;
        data2 = dcm.CurveData_2;
        data3 = dcm.CurveData_4;
        data4 = dcm.CurveData_6;
        data5 = dcm.CurveData_8;
        data6 = dcm.CurveData_A;
        data7 = dcm.CurveData_C;
        data8 = dcm.CurveData_E;
        data9 = dcm.CurveData_10;
        data10 = dcm.CurveData_12;
        data11 = dcm.CurveData_14;
        data12 = dcm.CurveData_16;
        data13 = dcm.CurveData_18;
        data = {data1 data2 data3 data4 data5 data6 data7 data8 data9 data10 data11 data12 data13};
    catch
        copyfile(path,[unlabed_path,labeler,'_',uid,'.dcm']);
        result = -2;
        return;
    end
    copyfile(path,[dicom_path,labeler,'_',uid,'.dcm']);
    img = double(img);
    img = (img-window_center+0.5*window_width)/window_width*255;
    img = uint8(img);
    image=figure('visible','off');
    imshow(img,'border','tight');%显示原图图像
    %saveas(image,[jpg1_path,labeler,'_',uid,'.jpg']);
    hold on;
    imwrite(mat2gray(img),[jpg1_path,labeler,'_haha',uid,'.jpg']) % Add by Boss Liu, save the image with origin sizes
    str_txt = [];
    for j = 1:1:13
        data_current = data{j};
        if size(data_current,1)==12
            for i = 4:-1:1
                plot(data_current(i*2-1),data_current(i*2),'ro','linewidth',2);
                str_txt = [num2str(data_current(i*2-1)),' ',num2str(data_current(i*2)),' ',str_txt];
            end
        elseif size(data_current,1)==2
            plot(data_current(1),data_current(2),'yo','linewidth',2);
            str_txt = [str_txt,num2str(data_current(1)),' ',num2str(data_current(2)),' '];
        else
            result = -3;
            return;
        end
    end
    saveas(image,[jpg2_path,labeler,'_',uid,'.jpg']);
    f_txt = fopen([txt_path,labeler,'_',uid,'.txt'],'a');
    fprintf(f_txt,'%s\r\n',str_txt);
    fclose(f_txt);
    result = 0;
    close all;
%%line([dcm5.CurveData_0(1) dcm5.CurveData_0(2)],[dcm5.CurveData_0(3) dcm5.CurveData_0(4)],'color','r','LineWidth',5);
% for i = 1:1:6
%     plot(dcm5.CurveData_0(i*2-1),dcm5.CurveData_0(i*2),'ro');
% end
%  for i = 1:1:4
%      plot(dcm6.CurveData_2(i*2-1),dcm6.CurveData_2(i*2),'bo');
%      plot(dcm6.CurveData_4(i*2-1),dcm6.CurveData_4(i*2),'bo');
%  end
%  plot(dcm6.CurveData_6(1),dcm6.CurveData_6(2),'yo');
%  plot(dcm6.CurveData_8(1),dcm6.CurveData_8(2),'yo');

end
