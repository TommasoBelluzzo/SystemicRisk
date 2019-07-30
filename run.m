warning('off','all');

close('all');
clearvars();
clc();
delete(allchild(0));

[path_base,~,~] = fileparts(mfilename('fullpath'));

if (~strcmpi(path_base(end),filesep()))
    path_base = [path_base filesep()];
end

if (~isempty(regexpi(path_base,'Editor')))
    path_base_fs = dir(path_base);
    is_live = ~all(cellfun(@isempty,regexpi({path_base_fs.name},'LiveEditorEvaluationHelper')));

    if (is_live)
        while (true)
            ia = inputdlg('It looks like the program is being executed as a live script. Please, manually enter the root folder of this package:','Manual Input Required');
    
            if (isempty(ia))
                return;
            end
            
            path_base_new = ia{:};

            if (isempty(path_base_new) || strcmp(path_base_new,path_base) || strcmp(path_base_new(1:end-1),path_base) || ~exist(path_base_new,'dir'))
               continue;
            end
            
            path_base = path_base_new;
            
            break;
        end
    end
end

if (~strcmpi(path_base(end),filesep()))
    path_base = [path_base filesep()];
end

paths_base = genpath(path_base);
paths_base = strsplit(paths_base,';');

for i = numel(paths_base):-1:1
    path_cur = paths_base{i};

    if (~strcmp(path_cur,path_base) && isempty(regexpi(path_cur,[filesep() 'Scripts'])))
        paths_base(i) = [];
    end
end

paths_base = [strjoin(paths_base,';') ';'];
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
