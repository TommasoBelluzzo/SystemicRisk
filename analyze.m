%% INITIALIZATION

up = false;
initialize;

ev = {'ds_dir' 'ds_version' 'out_dir' 'out_name' 'path_base' 'sn' 'temp_dir' 'temp_name'};
ev_len = numel(ev);

failures = false(ev_len,1);

for i = 1:numel(ev)
    ev_i = ev{i};

    if (exist(ev_i,'var') == 0)
        failures(i) = true;
    end

    eval(['failures(i) = isempty(' ev_i ') || ~ischar(' ev_i ');']);
end

if (any(failures))
    error('The initialization process failed.');
end

%% EXECUTION

target_sn = '';

files_path = fullfile(path_base,out_dir);
files = dir(fullfile(files_path,'*.mat'));
files_len = numel(files);

if (files_len == 0)
    warning('MATLAB:SystemicRisk','The analysis of systemic risk measures cannot be performed because no result files have been found.');
else
    v = cell(files_len,5);
    voff = 1;

    for i = 1:files_len
        file = files(i);
        file_path = fullfile(files_path,file.name);

        try
            s = load(file_path);
        catch e
            warning('MATLAB:SystemicRisk',['The result file ''' clear_text(file_path) ''' could not be loaded.' new_line() e.message]);
            continue;
        end

        if (~isstruct(s))
            warning('MATLAB:SystemicRisk',['The content of the result file ''' clear_text(file_path) ''' is invalid: output is not a structure.']);
            continue;
        end

        s_fields = fieldnames(s);

        if (numel(s_fields) ~= 1)
            warning('MATLAB:SystemicRisk',['The content of the result file ''' clear_text(file_path) ''' is invalid: structure fields not equal to 1.']);
            continue;
        end

        s_field = s_fields{1};

        if (isempty(regexpi(s_field,'^result_[a-z_]+$')))
            warning('MATLAB:SystemicRisk',['The content of the result file ''' clear_text(file_path) ''' is invalid: wrong naming convention of the structure field.']);
            continue;
        end

        s_value = s.(s_field);

        try
            s_value = validate_dataset(s_value);
        catch e
            warning('MATLAB:SystemicRisk',['The content of the result file ''' clear_text(file_path) ''' is invalid: inner data is not as expected.']);
            continue;
        end

        if (~isempty(target_sn) && ~strcmp(s_value.ResultSerial,target_sn))
            continue;    
        end

        v(voff,:) = {file_path s_value s_value.Result s_value.ResultDate s_value.ResultSerial};
        voff = voff + 1;
    end

    v(voff:end,:) = [];
    v_len = size(v,1);

    if (v_len == 0)
        warning('MATLAB:SystemicRisk','The analysis of systemic risk measures cannot be performed because no valid result files have been detected.');
        clearvars();
        return;
    end

    v = sortrows(v,[3 -4]);

    comparison_indices = strcmp(v(:,3),'Comparison');
    v = [v(~comparison_indices,:); v(comparison_indices,:)];

    [u,u_indices] = unique(v(:,3),'stable');
    u_len = numel(u);

    if (u_len ~= v_len)
        v = v(u_indices,:);
        warning('MATLAB:SystemicRisk','Multiple result files belonging to the same category have been found: only the most recent result file for each category will be analyzed.');
    end

    comparison_indices = strcmp(v(:,3),'Comparison');

    if (any(comparison_indices) && (numel(unique(v(:,5))) > 1))
        v(comparison_indices,:) = [];
        warning('MATLAB:SystemicRisk','Multiple result sets have been found: comparison will not be displayed.');
    end

    v_len = size(v,1);

    for i = 1:v_len
        v_i = v{i,2};

        try
            v_i.ResultAnalysis(v_i);
        catch e
            warning('MATLAB:SystemicRisk',['The result file ''' clear_text(v{i,1}) ''' could not be analyzed.' new_line() e.message]);
            continue;
        end
    end

end

%% CLEANUP

clearvars();
