%% VERSION CHECK

if (verLessThan('MATLAB','8.4'))
    error('The minimum required Matlab version is R2014b.');
end

%% CLEANUP

warning('off','all');
warning('on','MATLAB:SystemicRisk');

clc();

close('all','force');
delete(allchild(0));

clearvars('-except','up');

rng('default');

if (exist('up','var') == 0)
    up = true;
end

delete(gcp('nocreate'));

%% PARALLEL COMPUTING

if (up)
    pdprofile = parallel.defaultClusterProfile;
    pcluster = parcluster(pdprofile);
    delete(pcluster.Jobs);

    parpool(pcluster,'SpmdEnabled',false);
    pctRunOnAll warning('off','all');
    pctRunOnAll warning('on','MATLAB:SystemicRisk');
end

%% PATHS

[path_base,~,~] = fileparts(mfilename('fullpath'));

if (~strcmpi(path_base(end),filesep()))
    path_base = [path_base filesep()];
end

if (~isempty(regexpi(path_base,'Editor','once')))
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

    if (~strcmp(path_current,path_base) && isempty(regexpi(path_current,[filesep() 'Scripts'],'once')))
        paths_base(i) = [];
    end
end

paths_base = [strjoin(paths_base,';') ';'];
addpath(paths_base);

%% ENVIRONMENT VARIABLES

ds_dir = 'Datasets';
ds_version = 'v3.7';

try
    sn = ['INIT-' upper(char(java.util.UUID.randomUUID()))];
catch
    sn = randi([0 10000000]);
    sn = ['INIT-' sprintf('%08s',num2str(sn))];
end

temp_dir = 'Templates';
temp_name = 'Template';

out_dir = 'Results';
out_name = 'Result';

%% CLEANUP

clearvars('-except','ds_dir','ds_version','out_dir','out_name','path_base','sn','temp_dir','temp_name');
