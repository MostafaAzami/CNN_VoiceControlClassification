folder= 'Sv';
path_base =strcat( 'E:\\2-voice\\implementation\\DB\\',folder) ;
path_base =strcat( path,'\\img') ;

dirinfo = dir(path_base);
dirinfo([dirinfo.isdir]) = [];  %remove directories

for K = 1 : length(dirinfo)
    filename = dirinfo(K).name;
    path =strcat( path_base,filename) ;

    RGB = imread(path);
    gray = rgb2gray(RGB);
    %to=[1-115:792-875] sefid
    gray = gray(50:584,116:791);
    x=sum(gray>20,1);
    index= find(x>5);
    from = index(3);
    to= index(size(index,2));
    gray = gray(:,from:to);

    x=sum(gray>20,2);
    index= find(x>5);
    %[1:49, 585:675] sefid
    from = index(3);
    to= index(size(index,2));
    gray = gray(from:to,:);
    gray = imresize(gray,[80,80]);
    %imshow(gray)
    %saveas(gca,filename,'jpg')
    imwrite(gray, filename);

end

