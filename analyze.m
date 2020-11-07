%% INITIALIZATION

up = false;
initialize;

if ((exist('ds_version','var') == 0) || (exist('path_base','var') == 0))
    error('The initialization process failed.');
end

%% EXECUTION

files = dir(fullfile(path_base,['Results' filesep() '*.mat']));
files_len = numel(files);

if (files_len == 0)
    warning('MATLAB:SystemicRisk','The analysis of systemic risk measures cannot be performed because no result files have been found.');
else
    v = cell(files_len,3);
    voff = 1;
    
    for i = 1:files_len
        file = files(i);
        file_path = fullfile(file.folder,file.name);

        try
            s = load(file_path);
        catch e
            file_path_w = strrep(file_path,filesep(),[filesep() filesep()]);
            warning('MATLAB:SystemicRisk',['The result file ''' file_path_w ''' could not be loaded.' newline() e.message]);
            continue;
        end
        
        if (~isstruct(s))
            file_path_w = strrep(file_path,filesep(),[filesep() filesep()]);
            warning('MATLAB:SystemicRisk',['The content of the result file ''' file_path_w ''' is invalid: output is not a structure.']);
            continue;
        end
        
        s_fields = fieldnames(s);
        
        if (numel(s_fields) ~= 1)
            file_path_w = strrep(file_path,filesep(),[filesep() filesep()]);
            warning('MATLAB:SystemicRisk',['The content of the result file ''' file_path_w ''' is invalid: structure fields not equal to 1.']);
            continue;
        end
        
        s_field = s_fields{1};

        if (isempty(regexpi(s_field,'^result_[a-z_]+$')))
            file_path_w = strrep(file_path,filesep(),[filesep() filesep()]);
            warning('MATLAB:SystemicRisk',['The content of the result file ''' file_path_w ''' is invalid: wrong naming convention of the structure field.']);
            continue;
        end
        
        s_value = s.(s_field);
        
        try
            s_value = validate_dataset(s_value);
        catch e
            file_path_w = strrep(file_path,filesep(),[filesep() filesep()]);
            warning('MATLAB:SystemicRisk',['The content of the result file ''' file_path_w ''' is invalid: inner data is not as expected.']);
            continue;
        end

        result = s_value.Result;
        
        if (strcmp(result,'Comparison'))
            file_path_w = strrep(file_path,filesep(),[filesep() filesep()]);
            warning('MATLAB:SystemicRisk',['The content of the result file ''' file_path_w ''' is invalid: inner data is not supported.']);
            continue;
        end
        
        result_date = s_value.ResultDate;
        result_analysis = s_value.ResultAnalysis;

        v(voff,:) = {s_value result result_date};
        voff = voff + 1;
    end
    
    v(voff:end,:) = [];
    v_len = size(v,1);

    if (v_len == 0)
        warning('MATLAB:SystemicRisk','The analysis of systemic risk measures cannot be performed because no result files have been loaded.');
    else
        v = sortrows(v,[2 -3]);

        [u,u_indices] = unique(v(:,2));
        u_len = numel(u);
        
        if (u_len ~= v_len)
            v = v(u_indices,:);
            v_len = u_len;
            
            warning('MATLAB:SystemicRisk','Multiple result files belonging to the same category have been found: only the most recent result file for each category will be analyzed.');
        end
        
        for i = 1:v_len
            v_i = v{i};
            
            try
                v_i.ResultAnalysis(v_i);
            catch e
                file_path_w = strrep(file_path,filesep(),[filesep() filesep()]);
                warning('MATLAB:SystemicRisk',['The result file ''' file_path_w ''' could not be analyzed.' newline() e.message]);
                continue;
            end
        end 
    end
end

%% CLEANUP

clearvars();
