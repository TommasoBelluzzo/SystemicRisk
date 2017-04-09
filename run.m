warning off all;

close all;
clearvars;
clc;

paths = genpath(pwd);
addpath(paths);

data = parse_dataset('dataset.xlsx');

main_pro(data,fullfile(pwd,'results_pro.xlsx'),0.95,0.08,true);
main_net(data,fullfile(pwd,'results_net.xlsx'),0.05,true,true);

save('data.mat','data');

rmpath(paths);