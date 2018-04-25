function [ result ] = processDicom( path,labeler,window_width,window_center )
%function [ result ] = processDicom( path,labeler,dicom_path,txt_path,jpg1_path,jpg2_path,window_width,window_center )
%   此处显示详细说明
    img = dicomread(path);   %读取图像
	dcm = dicominfo(path);%存储信息
    cur_path = [pwd,'\'];
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
    
    uid = dcm.SOPInstanceUID;
    try
        data1 = dcm.CurveData_0;
        data2 = dcm.CurveData_2;
        data3 = dcm.CurveData_4;
        data = {data1 data2 data3};
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
    saveas(image,[jpg1_path,labeler,'_',uid,'.jpg']);
    hold on;
    
    str_txt = [];
    for j = 1:1:3
        data_current = data{j};
        if size(data_current,1)==12
            for i = 4:-1:1
                plot(data_current(i*2-1),data_current(i*2),'ro','linewidth',2);
                str_txt = [num2str(data_current(i*2-1)),' ',num2str(data_current(i*2)),' ',str_txt];
            end
        elseif size(data_current,1)==8
            for i = 1:2:5
                line([data_current(i),data_current(i+2)],[data_current(i+1),data_current(i+3)],'linewidth',2,'color','b');
            end
            line([data_current(1),data_current(7)],[data_current(2),data_current(8)],'linewidth',2,'color','b');
            str_txt = [str_txt,num2str(data_current(1)),' ',num2str(data_current(2)),' ',num2str(data_current(5)),' ',num2str(data_current(6)),' '];
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
