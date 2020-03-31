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

binary_version = 1.0;

file = fullfile(path_base,['Datasets' filesep() 'Example_Large.xlsx']);
[file_path,file_name,file_extension] = fileparts(file);

if (exist(file,'file') == 0)
	error(['The dataset file ''' file ''' could not be found.']);
end

mat = fullfile(file_path,[file_name '.mat']);

if (exist(mat,'file') == 2)
    file_dir = dir(file);
    file_lmd = datetime(file_dir.datenum,'ConvertFrom','datenum');
    mat_dir = dir(mat);
    mat_lmd = datetime(mat_dir.datenum,'ConvertFrom','datenum');
    
    if (file_lmd > mat_lmd)
        dataset_process = true;
    else
        dataset_process = false;
    end
else
    dataset_process = true;
end

if (dataset_process)
    data = parse_dataset(file,'dd/MM/yyyy','QQ yyyy','P',3);
    save(mat,'data');
    analyze_dataset(data);
else
    load(mat);
    
    if (data.BinaryVersion ~= binary_version)
        data = parse_dataset(file,'dd/MM/yyyy','QQ yyyy','P',3);
        save(mat,'data');
        analyze_dataset(data);
    end
end

%% MEASURES

setup = {
    % NAME            ENABLED  ANALYZE  FUNCTION
    'CrossSectional'  true     true     @(data,temp,file,analysis)run_cross_sectional(data,temp,file,0.95,0.40,0.08,0.40,analysis);
    'Default'         true     true     @(data,temp,file,analysis)run_default(data,temp,file,252,0.4,0.6,0.08,'BSM',0.95,analysis);
    'Connectedness'   true     true     @(data,temp,file,analysis)run_connectedness(data,temp,file,252,0.05,false,0.06,analysis);
    'Spillover'       true     true     @(data,temp,file,analysis)run_spillover(data,temp,file,252,10,2,4,'G',analysis);
    'Component'       true     true     @(data,temp,file,analysis)run_component(data,temp,file,252,0.99,0.2,0.75,analysis);
};

for i = 1:size(setup,1)
    category = setup{i,1};
    enabled = setup{i,2};
    analysis = setup{i,3};
    run_function = setup{i,4};
    
    if (~enabled)
        continue;
    end
    
    if (~data.(['Supports' category]))
        continue;
    end

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
