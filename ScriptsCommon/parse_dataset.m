% [INPUT]
% file = A string representing the full path to the Excel spreadsheet containing the dataset.
%
% [OUTPUT]
% data = A structure containing the parsed dataset.

function data = parse_dataset(varargin)

    persistent p;

    if (isempty(p))
        p = inputParser();
        p.addRequired('file',@(x)validateattributes(x,{'char'},{'nonempty','size',[1,NaN]}));
    end
    
    p.parse(varargin{:});
    res = p.Results;

    data = parse_dataset_internal(res.file);

end

function data = parse_dataset_internal(file)

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
    end
    
    if (shts_len > 5)
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
    
    rets = parse_table(file,1,'Returns');
    
    if (width(rets ) < 5)
        error('The dataset must must contain at least the following series: observations dates, benchmark returns and the returns of three firms to analyze.');
    end
    
    t = height(rets);

    if (t < 252)
        error('The dataset must contain at least 252 observations in order to run consistent calculations.');
    end

    if (any(ismissing(rets)))
        error('The ''Returns'' table contains invalid or missing values.');
    end

    dates_str = cellstr(datestr(rets{:,1},'dd/mm/yyyy'));
    dates_num = datenum(rets{:,1});
    dates_beg = dates_num(1);
    dates_end = dates_num(end);
    rets.Date = [];

    idx_ret = rets{:,1};
    idx_nam = rets.Properties.VariableNames{1};
    frms = numel(rets.Properties.VariableNames) - 1;
    frms_nam = rets.Properties.VariableNames(2:end);
    frms_ret = rets{:,2:end};

    frms_cap = parse_table(file,2,'Market Capitalization');

    if (any(ismissing(frms_cap)))
        error('The ''Market Capitalization'' table contains invalid or missing values.');
    end

    if ((datenum(frms_cap.Date(1)) >=  dates_beg) || (datenum(frms_cap.Date(end)) ~=  dates_end) || ((size(frms_cap,1) - 1) ~= t))
        error('The ''Returns'' table and the ''Market Capitalization'' table are mismatching.');
    end

    frms_cap.Date = [];
    
    if (~isequal(frms_cap.Properties.VariableNames,frms_nam))
        error('The ''Returns'' table and the ''Market Capitalization'' table are mismatching.');
    end
    
    frms_lia = parse_table(file,3,'Total Liabilities');
    
    if (any(ismissing(frms_lia)))
        error('The ''Total Liabilities'' table contains invalid or missing values.');
    end

    if ((datenum(frms_lia.Date(1)) ~=  dates_beg) || (datenum(frms_lia.Date(end)) ~=  dates_end) || (size(frms_lia,1) ~= t))
        error('The ''Returns'' table and the ''Total Liabilities'' table are mismatching.');
    end

    frms_lia.Date = [];
    
    if (~isequal(frms_lia.Properties.VariableNames,frms_nam))
        error('The ''Returns'' table and the ''Total Liabilities'' table are mismatching.');
    end

    if (stvars_off ~= -1)
        stvars = parse_table(file,stvars_off,'State Variables');
        
        if (any(ismissing(stvars)))
            error('The ''State Variables'' table contains invalid or missing values.');
        end
        
        if ((datenum(stvars.Date(1)) >=  dates_beg) || (datenum(stvars.Date(end)) ~=  dates_end) || ((size(stvars,1) - 1) ~= t))
            error('The ''Returns'' table and the ''State Variables'' table are mismatching.');
        end

        stvars.Date = [];
        stvars_lag = stvars{1:end-1,:};
    else
        stvars_lag = [];
    end
        
    if (grps_off ~= -1)
        grps = parse_table(file,grps_off,'Groups');
        
        if (~isequal(grps.Properties.VariableNames,{'Delimiter' 'Name'}))
            error('The ''Groups'' table contains invalid (wrong name) or misplaced (wrong order) columns.');
        end

        if (size(grps,1) < 2)
            error('In the ''Groups'' table, the number of rows must be greater than or equal to 2.');
        end

        grps_del = grps{:,1};

        if (grps_del(1) < 1)
            error('In the ''Groups'' table, the first group delimiter must be greater than or equal to 1.');
        end

        if (grps_del(end-1) >= frms)
            error('In the ''Groups'' table, the penultimate group delimiter must be less than the number of firms.');
        end

        if (~isnan(grps_del(end)))
            error('In the ''Groups'' table, the last group delimiter must be NaN.');
        end

        grps_del = grps_del(1:end-1);
        grps_nam = strtrim(grps{:,2});
    else
        grps_del = [];
        grps_nam = [];     
    end

    data = struct();
    data.DatesNum = dates_num;
    data.DatesStr = dates_str;
    data.Frms = frms;
    data.FrmsCap = frms_cap{2:end,:};
    data.FrmsCapLag = frms_cap{1:end-1,:};
    data.FrmsLia = frms_lia{:,:};
    data.FrmsNam = frms_nam;
    data.FrmsRet = frms_ret;
    data.Grps = length(grps_nam);
    data.GrpsDel = grps_del;
    data.GrpsNam = grps_nam;
    data.IdxNam = idx_nam;
    data.IdxRet = idx_ret;
    data.Obs = t;
    data.StVarsLag = stvars_lag;

end

function res = parse_table(file,sht,name)

    if (verLessThan('Matlab','9.1'))
        res = readtable(file,'Sheet',sht);
        
        if (any(~isempty(regexp(res.Properties.VariableNames,'^Var\d+','once'))))
            error(['The ''' name ''' table contains unnamed columns.']);
        end

        if (strcmp(name,'Groups'))
            res_vars = varfun(@class,res,'OutputFormat','cell');
            
            if (~strcmp(res_vars{1},'double') || ~strcmp(res_vars{2},'cell'))
                error(['The ''' name ''' table contains invalid or missing values.']);
            end
        else
            if (~strcmp(res.Properties.VariableNames(1),'Date'))
                error(['The first column of the ''' name ''' table must be called ''Date'' and must contain the time series dates.']);
            end
            
            res.Date = datetime(res.Date,'InputFormat','dd/MM/yyyy');
            
            res_vars = varfun(@class,res,'OutputFormat','cell');
            
            if (~all(strcmp(res_vars(2:end),'double')))
                error(['The ''' name ''' table contains invalid or missing values.']);
            end
        end
    else
        opts = detectImportOptions(file,'Sheet',sht);
        
        if (any(~isempty(regexp(opts.VariableNames,'^Var\d+$','once'))))
            error(['The ''' name ''' table contains unnamed columns.']);
        end

        if (strcmp(name,'Groups'))
            opts = setvartype(opts,{'double' 'char'});
        else
            if (~strcmp(opts.VariableNames(1),'Date'))
                error(['The first column of the ''' name ''' table must be called ''Date'' and must contain the time series dates.']);
            end

            opts = setvartype(opts,[{'datetime'} repmat({'double'},1,numel(opts.VariableNames)-1)]);
            opts = setvaropts(opts,'Date','InputFormat','dd/MM/yyyy');
        end

        res = readtable(file,opts);
    end

end
