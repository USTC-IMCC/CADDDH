mainfile = uigetdir
maindir = dir(mainfile);

len = length(maindir);
k=1;
while(k<=len)
    if(isequal(maindir(k).name,'.')||... % 去除系统自带的两个隐文件夹
          isequal(maindir(k).name,'..'))
        k=k+1;
        continue;
    end
    userfile = fullfile(mainfile,'\',maindir(k).name);
    if(isdir(userfile))
        userid = maindir(k).name;
        %fprintf('%s\n',userid);
        finddicom(userfile,userid);
        k = k + 1;
        continue;
    end
    k = k+1;
end
h=msgbox('成功运行','程序结束');

