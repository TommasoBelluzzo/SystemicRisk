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

    if (ispc() && verLessThan('Matlab','9.6'))
        [file_stat,file_shts,file_fmt] = xlsfinfo(file);
        
        if (isempty(file_stat) || ~strcmp(file_fmt,'xlOpenXMLWorkbook'))
            error('The dataset file is not a valid Excel spreadsheet.');
        end
    else
        [file_stat,file_shts] = xlsfinfo(file);
        
        if (isempty(file_stat))
            error('The dataset file is not a valid Excel spreadsheet.');
        end
    end

    file_shts = strtrim(file_shts);
    file_shts_oth = file_shts(2:end);
    
    if (~strcmp(file_shts{1},'Returns'))
        error('The first sheet of the dataset file must be the ''Returns'' one.');
    end
    
    if (~isempty(setdiff(file_shts_oth,{'Market Capitalization' 'Total Liabilities' 'State Variables' 'Groups'})))
        error('The dataset file contains unnecessary and/or unsupported sheets.');
    end

    cap_def = any(strcmp(file_shts_oth,{'Market Capitalization'}));
    lia_def = any(strcmp(file_shts_oth,{'Total Liabilities'}));
    
    if (cap_def && lia_def)
        full = true;
    else
        if (cap_def && ~lia_def)
            error('The dataset file contains the ''Market Capitalization'' sheet but not the ''Total Liabilities'' sheet.');
        elseif (~cap_def && lia_def)
            error('The dataset file contains the ''Total Liabilities'' sheet but not the ''Market Capitalization'' sheet.');
        end
        
        full = false;
    end

    try
        datetime('now','InputFormat',df);
    catch
        error('The specified date format is invalid.');
    end

    tab_rets = parse_table(file,1,'Returns',df);
    
    if (any(ismissing(tab_rets)))
        error('The ''Returns'' table contains invalid or missing values.');
    end
    
    if (width(tab_rets) < 5)
        error('The dataset must contain at least the following series: observations dates, benchmark returns and the returns of 3 firms to analyze.');
    end
    
    t = height(tab_rets);

    if (t < 253)
        error('The dataset must contain at least 253 observations (a full business year plus an additional observation at the beginning) in order to run consistent calculations.');
    end

    dates_str = cellstr(datetime(tab_rets{:,1},'InputFormat',df));
    dates_num = datenum(tab_rets{:,1});
    tab_rets.Date = [];

    idx_ret = tab_rets{2:end,1};
    idx_nam = tab_rets.Properties.VariableNames{1};
    frms = numel(tab_rets.Properties.VariableNames) - 1;
    frms_nam = tab_rets.Properties.VariableNames(2:end);
    frms_ret = tab_rets{2:end,2:end};

    for tab = {'Market Capitalization' 'Total Liabilities' 'State Variables' 'Groups'}

        tab_idx = find(strcmp(file_shts_oth,tab),1);

        switch (char(tab))

            case 'Market Capitalization'

                if (~isempty(tab_idx))
                    tab_cap = parse_table(file,tab_idx+1,'Market Capitalization',df);

                    if (any(ismissing(tab_cap)))
                        error('The ''Market Capitalization'' sheet contains invalid or missing values.');
                    end

                    if ((size(tab_cap,1) ~= t) || any(datenum(tab_cap.Date) ~= dates_num))
                        error('The observation dates in ''Returns'' and ''Market Capitalization'' sheets are mismatching.');
                    end

                    tab_cap.Date = [];

                    if (~isequal(tab_cap.Properties.VariableNames,frms_nam))
                        error('The firm names in ''Returns'' and ''Market Capitalization'' sheets are mismatching.');
                    end
                    
                    tc = tab_cap;
                    tab_cap = tc{2:end,:};
                    tab_cap_lag = tc{1:end-1,:};
                else
                    tab_cap = [];
                    tab_cap_lag = [];
                end

            case 'Total Liabilities'
                
                if (~isempty(tab_idx))
                    tab_lia = parse_table(file,tab_idx+1,'Total Liabilities',df);

                    if (any(ismissing(tab_lia)))
                        error('The ''Total Liabilities'' sheet contains invalid or missing values.');
                    end

                    if ((size(tab_lia,1) ~= t) || any(datenum(tab_lia.Date) ~= dates_num))
                        error('The observation dates in ''Returns'' and ''Total Liabilities'' sheets are mismatching.');
                    end

                    tab_lia.Date = [];

                    if (~isequal(tab_lia.Properties.VariableNames,frms_nam))
                        error('The firm names in ''Returns'' and ''Total Liabilities'' sheets are mismatching.');
                    end
                    
                    tab_lia = tab_lia{2:end,:};
                else
                    tab_lia = [];
                end

            case 'State Variables'

                if (~isempty(tab_idx))
                    stvars = parse_table(file,tab_idx+1,'State Variables',df);

                    if (any(ismissing(stvars)))
                        error('The ''State Variables'' sheet contains invalid or missing values.');
                    end

                    if ((size(stvars,1) ~= t) || any(datenum(stvars.Date) ~= dates_num))
                        error('The observation dates of ''Returns'' and ''State Variables'' sheets are mismatching.');
                    end

                    stvars.Date = [];
                    stvars_lag = stvars{1:end-1,:};
                else
                    stvars_lag = [];
                end

            case 'Groups'
                
                if (~isempty(tab_idx))
                    grps = parse_table(file,tab_idx+1,'Groups',df);

                    if (any(ismissing(grps)))
                        error('The ''Groups'' sheet contains invalid or missing values.');
                    end

                    if (~isequal(grps.Properties.VariableNames,{'Name' 'Count'}))
                        error('The ''Groups'' sheet contains invalid (wrong name) or misplaced (wrong order) columns.');
                    end

                    if (size(grps,1) < 2)
                        error('In the ''Groups'' sheet, the number of rows must be greater than or equal to 2.');
                    end

                    grps_cnt = grps{:,2};

                    if (any(grps_cnt <= 0))
                        error('The ''Groups'' sheet contains one or more groups with an invalid number of firms.');
                    end

                    if (sum(grps_cnt) ~= frms)
                        error('In the ''Groups'' sheet, the number of firms must be equal to the one defined in the ''Returns'' sheet.');
                    end

                    grps_del = cumsum(grps_cnt(1:end-1,:));
                    grps_nam = strtrim(grps{:,1});
                else
                    grps_del = [];    
                    grps_nam = [];
                end

        end
    end

    data = struct();
    data.DatesNum = dates_num(2:end);
    data.DatesStr = dates_str(2:end);
    data.Frms = frms;
    data.FrmsCap = tab_cap;
    data.FrmsCapLag = tab_cap_lag;
    data.FrmsLia = tab_lia;
    data.FrmsNam = frms_nam;
    data.FrmsRet = frms_ret;
    data.Full = full;
    data.Grps = numel(grps_nam);
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
