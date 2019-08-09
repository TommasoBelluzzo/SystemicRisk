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
        pwd_curr = pwd();

        if (~strcmpi(pwd_curr(end),filesep()))
            pwd_curr = [pwd_curr filesep()];
        end
        
        while (true) 
            ia = inputdlg('It looks like the program is being executed in a non-standard mode. Please, confirm or change the root folder of this package:','Manual Input Required',1,{pwd_curr});
    
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

file_tpro = fullfile(path_base,['Templates' filesep() 'TemplatePRO.xlsx']);
file_tnet = fullfile(path_base,['Templates' filesep() 'TemplateNET.xlsx']);

file_dset = fullfile(path_base,['Datasets' filesep() 'Example_Large.xlsx']);
file_rpro = fullfile(path_base,['Results' filesep() 'ResultsPRO.xlsx']);
file_rnet = fullfile(path_base,['Results' filesep() 'ResultsNET.xlsx']);
file_mat = fullfile(path_base,['Results' filesep() 'Data.mat']);

data = parse_dataset(file_dset);

main_pro(data,file_tpro,file_rpro,0.95,0.40,0.08,true);
pause(2);
main_net(data,file_tnet,file_rnet,0.05,true,true);

save(file_mat,'data');

rmpath(paths_base);
