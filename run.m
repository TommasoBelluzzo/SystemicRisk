%% VERSION CHECK

if (verLessThan('MATLAB','8.3'))
    error('The minimum required Matlab version is R2014a.');
end

%% CLEANUP

warning('off','all');
warning('on','MATLAB:SystemicRisk');

close('all');
clearvars();
clc();
delete(allchild(0));
delete(gcp('nocreate'));

%% INITIALIZATION

parpool('local','SpmdEnabled',false);
pctRunOnAll warning('off','all');
pctRunOnAll warning('on','MATLAB:SystemicRisk');

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
            answer = inputdlg('The script is being executed in live mode. Please, confirm or change its root folder:','Manual Input Required',1,{pwd_current});
    
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

%% DATASET

file = fullfile(path_base,['Datasets' filesep() 'Example_Large.xlsx']);
data = parse_dataset(file,'dd/MM/yyyy','QQ yyyy','P',3);

mat = fullfile(path_base,['Results' filesep() 'Data.mat']);
save(mat,'data');

analyze_dataset(data);

%% MEASURES

setup = {
    'CrossSectional' false true @(data,temp,file,analysis)run_cross_sectional(data,temp,file,0.95,0.40,0.08,0.40,analysis);
    'Default'        true true @(data,temp,file,analysis)run_default(data,temp,file,252,0.4,0.6,0.08,0.95,analysis);
    'Connectedness'  false true @(data,temp,file,analysis)run_connectedness(data,temp,file,252,0.05,true,0.06,analysis);
    'Spillover'      false true @(data,temp,file,analysis)run_spillover(data,temp,file,252,10,2,4,'G',analysis);
    'Component'      false true @(data,temp,file,analysis)run_component(data,temp,file,252,0.2,0.75,analysis);
};

for i = 1:size(setup,1)
    enabled = setup{i,2};
    
    if (enabled)
        category = setup{i,1};
        analysis = setup{i,3};
        run_function = setup{i,4};
        
        pause(2);

        temp = fullfile(path_base,['Templates' filesep() 'Template' category '.xlsx']);
        out = fullfile(path_base,['Results' filesep() 'Results' category '.xlsx']);
        [result,stopped] = run_function(data,temp,out,analysis);

        if (stopped)
            return;
        end
        
        category_reference = ['result' lower(regexprep(category,'([A-Z])','_$1'))];

        eval([category_reference ' = result;']);
        clear('result','stopped');

        mat = fullfile(path_base,['Results' filesep() 'Results' category '.mat']);
        save(mat,category_reference);
    end
end
