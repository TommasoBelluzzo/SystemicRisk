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

path_dset = strrep('Datasets\Example_Large.xlsx','\',filesep());

path_tpro = strrep('Templates\TemplatePRO.xlsx','\',filesep());
file_tpro = fullfile(path_base,path_tpro);
path_rpro = strrep('Results\ResultsPRO.xlsx','\',filesep());
file_rpro = fullfile(path_base,path_rpro);

path_tnet = strrep('Templates\TemplateNET.xlsx','\',filesep());
file_tnet = fullfile(path_base,path_tnet);
path_rnet = strrep('Results\ResultsNET.xlsx','\',filesep());
file_rnet = fullfile(path_base,path_rnet);

data = parse_dataset(fullfile(path_base,path_dset));

main_pro(data,file_tpro,file_rpro,0.95,0.40,0.08,true);
pause(2);
main_net(data,file_tnet,file_rnet,0.05,true,true);

save('data.mat','data');

rmpath(paths_base);