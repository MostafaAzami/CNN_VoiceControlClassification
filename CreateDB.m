folder= 'Sv';
path =strcat( 'E:\\2-voice\\implementation\\DB\\',folder) ;
path =strcat( path,'\\voice') ;

dirinfo = dir(path);
dirinfo([dirinfo.isdir]) = [];  %remove directories

for K = 1 : length(dirinfo)
    K
  filename = dirinfo(K).name;
  [y,fs] = audioread(strcat(path,'\\',filename));
  y=y(:,1);
  window = hamming(512);
  noverlap=256;
  nfft= 1024;
  [s,f,t,p]= spectrogram(y,window,noverlap, nfft, fs,'yaxis');
  surf(t,f,10*log10(p),'edgecolor','none'); 
  axis tight;
  view(0,90);
  colormap(hot);
  set(gca,'clim',[-80 -30]);
  
  xlim([0 3])
  set(gca,'Color','k')
  set(gcf, 'InvertHardCopy', 'off');

  xlabel('time s');
  fig=  ylabel('frequency khz');
  filename= strcat(folder,int2str(K));
  saveas(gca,filename,'jpg')

end