warning off all;

close all;
clearvars;
clc;

[path,~,~] = fileparts(mfilename('fullpath'));

paths = genpath(path);
addpath(paths);

data = parse_dataset('dataset.xlsx');

main_pro(data,fullfile(path,'results_pro.xlsx'),0.95,0.40,0.08,true);
main_net(data,fullfile(pwd,'results_net.xlsx'),0.05,true,true);

save('data.mat','data');

rmpath(paths);
