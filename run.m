warning('off','all');

close('all');
clearvars();
clc();
delete(allchild(0));

delete(gcp('nocreate'));
parpool('SpmdEnabled',false);
pctRunOnAll warning('off', 'all');

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

pause(1);

out_temp_stochastic = fullfile(path_base,['Templates' filesep() 'TemplateStochastic.xlsx']);
out_file_stochastic = fullfile(path_base,['Results' filesep() 'ResultsStochastic.xlsx']);
result_stochastic = run_stochastic(data,out_temp_stochastic,out_file_stochastic,0.95,0.40,0.08,0.40,true);
mat_stochastic = fullfile(path_base,['Results' filesep() 'DataStochastic.mat']);
save(mat_stochastic,'result_stochastic');

pause(2);

out_temp_connectedness = fullfile(path_base,['Templates' filesep() 'TemplateConnectedness.xlsx']);
out_file_connectedness = fullfile(path_base,['Results' filesep() 'ResultsConnectedness.xlsx']);
result_connectedness = run_connectedness(data,out_temp_connectedness,out_file_connectedness,252,0.05,true,0.06,true);
mat_connectedness = fullfile(path_base,['Results' filesep() 'DataConnectedness.mat']);
save(mat_connectedness,'result_connectedness');

pause(2);

out_temp_spillover = fullfile(path_base,['Templates' filesep() 'TemplateSpillover.xlsx']);
out_file_spillover = fullfile(path_base,['Results' filesep() 'ResultsSpillover.xlsx']);
result_spillover = run_spillover(data,out_temp_spillover,out_file_spillover,252,10,2,4,true,true);
mat_spillover = fullfile(path_base,['Results' filesep() 'DataSpillover.mat']);
save(mat_spillover,'result_spillover');

pause(1);

mat_dataset = fullfile(path_base,['Results' filesep() 'Dataset.mat']);
save(mat_dataset,'data');
