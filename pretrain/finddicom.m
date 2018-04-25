function [ result ] = finddicom( x,y )
%x是路径
%使用递归方法查找
%查找x目录下的文件，文件夹则调用自身，是该文件则打印路径\
    dirx = dir(x);
    for i=1:length(dirx)
        if(isequal(dirx(i).name,'.')||isequal(dirx(i).name,'..')) % 去除系统自带的两个隐文件夹
            continue;
        end
        subfile = fullfile(x,'\',dirx(i).name);
        if(isdir(subfile))
            finddicom(subfile,y);
            continue;
        end
        if(~isdir(subfile))
            if~(isequal(dirx(i).name,'DICOMDIR')||isequal(dirx(i).name,'LOCKFILE')||isequal(dirx(i).name,'VERSION'))
                fprintf('%s\t',subfile);
                fprintf('%s\n',y);
                processDicom(subfile,y);
                continue;
            end
        end
    end

end

