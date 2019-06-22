% [INPUT]
% file = A string representing the full path to the Excel spreadsheet containing the dataset.
% df   = A string representing the date format used in the Excel spreadsheet (optional, default=dd/MM/yyyy).
%
% [OUTPUT]
% data = A structure containing the parsed dataset.

function data = parse_dataset(varargin)

    persistent p;

    if (isempty(p))
        p = inputParser();
        p.addRequired('file',@(x)validateattributes(x,{'char'},{'nonempty','size',[1,NaN]}));
        p.addOptional('df','dd/MM/yyyy',@(x)validateattributes(x,{'char'},{'nonempty','size',[1,NaN]}));
    end

    p.parse(varargin{:});
    res = p.Results;

    data = parse_dataset_internal(res.file,res.df);

end

function data = parse_dataset_internal(file,df)

    if (exist(file,'file') == 0)
        error('The dataset file does not exist.');
    end

    [file_stat,file_shts,file_fmt] = xlsfinfo(file);

    if (isempty(file_stat) || (ispc() && ~strcmp(file_fmt,'xlOpenXMLWorkbook')))
        error('The dataset file is not a valid Excel spreadsheet.');
    end

    shts_len = length(file_shts);
    
    if (shts_len < 3)
        error('The dataset does not contain all the required sheets.');
    elseif (shts_len > 5)
        error('The dataset contains unnecessary sheets.');
    end
    
    file_shts = strtrim(file_shts);
    
    if (~isequal(file_shts(1:3),{'Returns' 'Market Capitalization' 'Total Liabilities'}))
        error('The dataset contains invalid (wrong name) or misplaced (wrong order) sheets.');
    end
    
    stvars_off = -1;
    grps_off = -1;
    
    if (shts_len == 4)
        file_shts_4 = file_shts{4};
        
        if (strcmp(file_shts_4,'State Variables'))
            stvars_off = 4;
        elseif (strcmp(file_shts_4,'Groups'))
            grps_off = 4;
        else
            error('The dataset contains invalid (wrong name) or misplaced (wrong order) sheets.');
        end
    elseif (shts_len == 5)
        if (~isequal(file_shts(4:5),{'State Variables' 'Groups'}))
            error('The dataset contains invalid (wrong name) or misplaced (wrong order) sheets.');
        end
        
        stvars_off = 4;
        grps_off = 5;
    end
    
    try
        datetime('now','InputFormat',df);
    catch
        error('The specified date format is invalid.');
    end

    rets = parse_table(file,1,'Returns',df);
    
    if (any(ismissing(rets)))
        error('The ''Returns'' table contains invalid or missing values.');
    end
    
    if (width(rets) < 5)
        error('The dataset must must contain at least the following series: observations dates, benchmark returns and the returns of 3 firms to analyze.');
    end
    
    t = height(rets);

    if (t < 253)
        error('The dataset must contain at least 253 observations (a full business year plus an additional observation at the beginning) in order to run consistent calculations.');
    end

    dates_str = cellstr(datetime(rets{:,1},'InputFormat',df));
    dates_num = datenum(rets{:,1});
    rets.Date = [];

    idx_ret = rets{2:end,1};
    idx_nam = rets.Properties.VariableNames{1};
    frms = numel(rets.Properties.VariableNames) - 1;
    frms_nam = rets.Properties.VariableNames(2:end);
    frms_ret = rets{2:end,2:end};

    frms_cap = parse_table(file,2,'Market Capitalization',df);

    if (any(ismissing(frms_cap)))
        error('The ''Market Capitalization'' table contains invalid or missing values.');
    end

    if ((size(frms_cap,1) ~= t) || any(datenum(frms_cap.Date) ~= dates_num))
        error('The observation dates of ''Returns'' table and ''Market Capitalization'' table are mismatching.');
    end
    
    frms_cap.Date = [];
    
    if (~isequal(frms_cap.Properties.VariableNames,frms_nam))
        error('The firm names of ''Returns'' table and ''Market Capitalization'' table are mismatching.');
    end

    frms_lia = parse_table(file,3,'Total Liabilities',df);
    
    if (any(ismissing(frms_lia)))
        error('The ''Total Liabilities'' table contains invalid or missing values.');
    end

    if ((size(frms_lia,1) ~= t) || any(datenum(frms_lia.Date) ~= dates_num))
        error('The observation dates of ''Returns'' table and ''Total Liabilities'' table are mismatching.');
    end
    
    frms_lia.Date = [];
    
    if (~isequal(frms_lia.Properties.VariableNames,frms_nam))
        error('The firm names of ''Returns'' table and ''Total Liabilities'' table are mismatching.');
    end

    if (stvars_off ~= -1)
        stvars = parse_table(file,stvars_off,'State Variables',df);
        
        if (any(ismissing(stvars)))
            error('The ''State Variables'' table contains invalid or missing values.');
        end
        
        if ((size(stvars,1) ~= t) || any(datenum(stvars.Date) ~= dates_num))
            error('The observation dates of ''Returns'' table and ''State Variables'' table are mismatching.');
        end

        stvars.Date = [];

        stvars_lag = stvars{1:end-1,:};
    else
        stvars_lag = [];
    end
        
    if (grps_off ~= -1)
        grps = parse_table(file,grps_off,'Groups',df);
        
        if (any(ismissing(stvars)))
            error('The ''Groups'' table contains invalid or missing values.');
        end
        
        if (~isequal(grps.Properties.VariableNames,{'Name' 'Count'}))
            error('The ''Groups'' table contains invalid (wrong name) or misplaced (wrong order) columns.');
        end

        if (size(grps,1) < 2)
            error('In the ''Groups'' table, the number of rows must be greater than or equal to 2.');
        end

        grps_cnt = grps{:,2};
        
        if (any(grps_cnt <= 0))
            error('The ''Groups'' table contains one or more groups with an invalid number of firms.');
        end
        
        if (sum(grps_cnt) ~= frms)
            error('In the ''Groups'' table, the number of firms must be equal to the one defined in the ''Returns'' table.');
        end

        grps_del = cumsum(grps_cnt(1:end-1,:));
        grps_nam = strtrim(grps{:,1});
    else
        grps_del = [];    
        grps_nam = [];
    end

    data = struct();
    data.DatesNum = dates_num(2:end);
    data.DatesStr = dates_str(2:end);
    data.Frms = frms;
    data.FrmsCap = frms_cap{2:end,:};
    data.FrmsCapLag = frms_cap{1:end-1,:};
    data.FrmsLia = frms_lia{2:end,:};
    data.FrmsNam = frms_nam;
    data.FrmsRet = frms_ret;
    data.Grps = length(grps_nam);
    data.GrpsDel = grps_del;
    data.GrpsNam = grps_nam;
    data.IdxNam = idx_nam;
    data.IdxRet = idx_ret;
    data.Obs = t - 1;
    data.StVarsLag = stvars_lag;

end

function res = parse_table(file,sht,name,df)

    if (verLessThan('Matlab','9.1'))
        res = readtable(file,'Sheet',sht);
        
        if (~all(cellfun(@isempty,regexp(res.Properties.VariableNames,'^Var\d+$','once'))))
            error(['The ''' name ''' table contains unnamed columns.']);
        end

        if (strcmp(name,'Groups'))
            res_vars = varfun(@class,res,'OutputFormat','cell');
            
            if (~strcmp(res_vars{1},'cell') || ~strcmp(res_vars{2},'double'))
                error(['The ''' name ''' table contains invalid or missing values.']);
            end
        else
            if (~strcmp(res.Properties.VariableNames(1),'Date'))
                error(['The first column of the ''' name ''' table must be called ''Date'' and must contain the observation dates.']);
            end
            
            res.Date = datetime(res.Date,'InputFormat',df);
            
            res_vars = varfun(@class,res,'OutputFormat','cell');
            
            if (~all(strcmp(res_vars(2:end),'double')))
                error(['The ''' name ''' table contains invalid or missing values.']);
            end
        end
    else
        opts = detectImportOptions(file,'Sheet',sht);
        
        if (~all(cellfun(@isempty,regexp(opts.VariableNames,'^Var\d+$','once'))))
            error(['The ''' name ''' table contains unnamed columns.']);
        end

        if (strcmp(name,'Groups'))
            opts = setvartype(opts,{'char' 'double'});
        else
            if (~strcmp(opts.VariableNames(1),'Date'))
                error(['The first column of the ''' name ''' table must be called ''Date'' and must contain the observation dates.']);
            end

            opts = setvartype(opts,[{'datetime'} repmat({'double'},1,numel(opts.VariableNames)-1)]);
            opts = setvaropts(opts,'Date','InputFormat',df);
        end

        res = readtable(file,opts);
    end

end
