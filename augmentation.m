folder= 'B';
path_base =strcat( 'E:\\2-voice\\implementation\\DB\\',folder) ;
path_base =strcat( path,'\\grayimg-orgsize') ;

dirinfo = dir(path_base);
dirinfo([dirinfo.isdir]) = [];  %remove directories

for K = 1 : length(dirinfo)
    filename = dirinfo(K).name;
    path =strcat( path_base,filename) ;

    gray = imread(path);
    len = size(gray,2);
    z= zeros(size(gray,1), 25);
    z1= zeros(size(gray,1), 40);
    graytemp= gray(:,1:len-25);
    gray1=[graytemp z];
    
    graytemp= gray(:,26:len);
    gray2=[z graytemp];
    
    graytemp= gray(:,1:len-40);
    gray3=[graytemp z1];
    
    graytemp= gray(:,41:len);
    gray4=[z1 graytemp];
    
    gray = imresize(gray,[80,80]);
    gray1 = imresize(gray1,[80,80]);
    gray2 = imresize(gray2,[80,80]);
    gray3 = imresize(gray3,[80,80]);
    gray4 = imresize(gray4,[80,80]);
    %imshow(gray)
    %saveas(gca,filename,'jpg')
    imwrite(gray, filename);
    imwrite(gray1, strcat('ag1_',filename));
    imwrite(gray2, strcat('ag2_',filename));
    imwrite(gray3, strcat('ag3_',filename));
    imwrite(gray4, strcat('ag4_',filename));

end

