mainfile = uigetdir
maindir = dir(mainfile);

dicomPath = 'E:\\髋关节\\SaveDicom';
textPath = 'E:\\髋关节\\Text';
jpg1Path = 'E:\髋关节\jpg1Path';
jpg2Path = 'E:\髋关节\jpg2Path';

len = length(maindir);
k=1;
while(k<=len)
    if(isequal(maindir(k).name,'.')||... % 去除系统自带的两个隐文件夹
          isequal(maindir(k).name,'..'))
        k=k+1;
        continue;
    end
    userfile = fullfile(mainfile,'\',maindir(k).name);
    if(isfolder(userfile))
        userid = maindir(k).name;
        %fprintf('%s\n',userid);
        finddicom(userfile,userid);
        k = k + 1;
        continue;
    end
    k = k+1;
end

function finddicom(x,y)
%x是路径
%使用递归方法查找
%查找x目录下的文件，文件夹则调用自身，是该文件则打印路径\
    dirx = dir(x);
    for i=1:length(dirx)
        if(isequal(dirx(i).name,'.')||... % 去除系统自带的两个隐文件夹
          isequal(dirx(i).name,'..'))
            continue;
        end
        subfile = fullfile(x,'\',dirx(i).name);
        if(isfolder(subfile))
            finddicom(subfile,y);
            continue;
        end
        if(~isfolder(subfile))
            if(isequal(dirx(i).name,'I1000000'))
                fprintf('%s\t',subfile);
                fprintf('%s\n',y);
                dicomPath = 'E:\\髋关节\\SaveDicom\\';
                textPath = 'E:\\髋关节\\Text\';
                jpg1Path = 'E:\髋关节\jpg1Path\';
                jpg2Path = 'E:\髋关节\jpg2Path\';
                processDicom(subfile,y,dicomPath,textPath,jpg1Path,jpg2Path);
                continue;
            end
        end
    end
end

