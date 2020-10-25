%% VERSION CHECK

if (verLessThan('MATLAB','8.4'))
    error('The minimum required Matlab version is R2014b.');
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

pdprofile = parallel.defaultClusterProfile;
pcluster = parcluster(pdprofile);
delete(pcluster.Jobs);

parpool(pcluster,'SpmdEnabled',false);
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

%% DATASET PARSING

ds_version = 'v3.1';
ds_process = false;

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
        ds_process = true;
    else
        load(mat);
        
        if ((exist('ds','var') == 0) || ~strcmp(ds.Version,ds_version))
            ds_process = true;
        end
    end
else
    ds_process = true;
end

if (ds_process)
    ds = parse_dataset(file,ds_version,'dd/mm/yyyy','QQ yyyy','P','R',0.05);
    analyze_dataset(ds);
    save(mat,'ds');
end

%% EXECUTION

bw = 252;

measures_setup = {
%   NAME                 ENABLED  ANALYZE  COMPARE  FUNCTION
    'Component'          true     true     true     @(ds,temp,file,analyze)run_component(ds,temp,file,bw,0.99,0.98,0.05,0.2,0.75,analyze);
    'Connectedness'      true     true     true     @(ds,temp,file,analyze)run_connectedness(ds,temp,file,bw,0.05,false,0.06,analyze);
    'CrossEntropy'       true     true     true     @(ds,temp,file,analyze)run_cross_entropy(ds,temp,file,bw,'G',0.4,'W','N',analyze);
    'CrossQuantilogram'  true     true     true     @(ds,temp,file,analyze)run_cross_quantilogram(ds,temp,file,bw,0.05,60,'SB',0.05,100,analyze);
    'CrossSectional'     true     true     true     @(ds,temp,file,analyze)run_cross_sectional(ds,temp,file,0.95,0.40,0.08,0.40,3,analyze);
    'Default'            true     true     true     @(ds,temp,file,analyze)run_default(ds,temp,file,bw,'BSM',3,0.08,0.45,2,0.10,100,5,0.95,analyze);
    'Liquidity'          true     true     true     @(ds,temp,file,analyze)run_liquidity(ds,temp,file,bw,21,5,'B',500,0.01,0.0004,analyze);
    'RegimeSwitching'    true     true     true     @(ds,temp,file,analyze)run_regime_switching(ds,temp,file,true,true,true,analyze);
    'Spillover'          true     true     true     @(ds,temp,file,analyze)run_spillover(ds,temp,file,bw,10,'G',2,4,analyze);
};

ml = repmat({''},1,100);
md = NaN(ds.T,100);
moff = 1;

for i = 1:size(measures_setup,1)
    [category,enabled,analyze,compare,run_function] = measures_setup{i,:};
    
    if (~enabled)
        continue;
    end
    
    if (~ds.(['Supports' category]))
        continue;
    end

    pause(2);

    temp = fullfile(path_base,['Templates' filesep() 'Template' category '.xlsx']);
    out = fullfile(path_base,['Results' filesep() 'Results' category '.xlsx']);
    [result,stopped] = run_function(ds,temp,out,analyze);

    if (stopped)
        clear('-regexp','(?!^(?:ds|result_[a-z_]+)$)^.+$');
        return;
    end
    
    if (compare && isfield(result,'ComparisonReferences') && ~isempty(result.ComparisonReferences))
        for j = 1:size(result.ComparisonReferences,1)
            [c_field,c_indices,c_labels] = result.ComparisonReferences{j,:};
            c_measures = result.(c_field);

            if (~isempty(c_indices))
                c_measures = c_measures(:,c_indices);
            end
            
            c_measures_len = size(c_measures,2);
            c_coffset = moff + c_measures_len - 1;

            ml(moff:c_coffset) = c_labels;
            md(:,moff:c_coffset) = c_measures;
            moff = moff + c_measures_len;
        end
    end

    category_reference = ['result' lower(regexprep(category,'([A-Z])','_$1'))];
    eval([category_reference ' = result;']);

    mat = fullfile(path_base,['Results' filesep() 'Results' category '.mat']);
    save(mat,category_reference);
end

ml(moff:end) = [];
md(:,moff:end) = [];

if (numel(ml) <= 1)
    enabled_all = [measures_setup{:,2}];
    
    if (any(enabled_all))
        compare_all = [measures_setup{:,4}];
        compare_all = compare_all(enabled_all);
        
        if (any(compare_all))
            warning('MATLAB:SystemicRisk','The comparison of systemic risk measures cannot be performed because the sample is empty or contains only one indicator.');
        end
    end
else
    pause(2);
    
    temp = fullfile(path_base,['Templates' filesep() 'TemplateComparison.xlsx']);
    out = fullfile(path_base,['Results' filesep() 'ResultsComparison.xlsx']);
    co = bw + 1;
    analyze = true;

    result_comparison = run_comparison(ds,temp,out,ml,md,co,[],21,'AIC',0.01,false,'GG',analyze);
end

clear('-regexp','(?!^(?:ds|result_[a-z_]+)$)^.+$');
