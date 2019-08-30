warning('off','all');

spmd
	warning('off','all')
end

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
        pwd_current = pwd();

        if (~strcmpi(pwd_current(end),filesep()))
            pwd_current = [pwd_current filesep()];
        end
        
        while (true) 
            answer = inputdlg('It looks like the program is being executed in a non-standard mode. Please, confirm or change the root folder of this package:','Manual Input Required',1,{pwd_current});
    
            if (isempty(answer))
                return;
            end
            
            path_base_new = answer{:};

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
    path_current = paths_base{i};

    if (~strcmp(path_current,path_base) && isempty(regexpi(path_current,[filesep() 'Scripts'])))
        paths_base(i) = [];
    end
end

paths_base = [strjoin(paths_base,';') ';'];
addpath(paths_base);

dataset = fullfile(path_base,['Datasets' filesep() 'Example_Large.xlsx']);
data = parse_dataset(dataset);

out_temp_stochastic = fullfile(path_base,['Templates' filesep() 'TemplateStochastic.xlsx']);
out_file_stochastic = fullfile(path_base,['Results' filesep() 'ResultsStochastic.xlsx']);
result_stochastic = run_stochastic(data,out_temp_stochastic,out_file_stochastic,0.95,0.40,0.08,0.40,true);
mat_stochastic = fullfile(path_base,['Results' filesep() 'DataStochastic.mat']);
save(mat_stochastic,'result_stochastic');

pause(2);

out_temp_network = fullfile(path_base,['Templates' filesep() 'TemplateNetwork.xlsx']);
out_file_network = fullfile(path_base,['Results' filesep() 'ResultsNetwork.xlsx']);
result_network = run_network(data,out_temp_network,out_file_network,252,0.05,true,0.06,2,4,true,true);
mat_network = fullfile(path_base,['Results' filesep() 'DataNetwork.mat']);
save(mat_network,'result_network');

mat_dataset = fullfile(path_base,['Results' filesep() 'Dataset.mat']);
save(mat_dataset,'data');
