warning('off','all');

close('all');
clearvars();
clc();

[path,~,~] = fileparts(mfilename('fullpath'));
paths = genpath(path);

addpath(paths);

data = parse_dataset(fullfile(path,'\Datasets\Short.xlsx'));

main_pro(data,fullfile(path,'\Results\ResultsPRO.xlsx'),0.95,0.40,0.08,true);
pause(5);
main_net(data,fullfile(path,'\Results\ResultsNET.xlsx'),0.05,true,true);

save('data.mat','data');

rmpath(paths);
