warning('off','all');

close('all');
clearvars();
clc();
delete(allchild(0));

[path_base,~,~] = fileparts(mfilename('fullpath'));

if (~strcmpi(path_base(end),filesep()))
    path_base = [path_base filesep()];
end

paths_base = genpath(path_base);
addpath(paths_base);

path_dset = strrep('Datasets\Example.xlsx','\',filesep());
path_rpro = strrep('Results\ResultsPRO.xlsx','\',filesep());
path_rnet = strrep('Results\ResultsNET.xlsx','\',filesep());

data = parse_dataset(fullfile(path_base,path_dset));

main_pro(data,fullfile(path_base,path_rpro),0.95,0.40,0.08,true);
pause(2);
main_net(data,fullfile(path_base,path_rnet),0.05,true,true);

save('data.mat','data');

rmpath(paths_base);