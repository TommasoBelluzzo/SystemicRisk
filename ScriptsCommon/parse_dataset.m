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

    if (isempty(file_stat) || ~strcmp(file_fmt,'xlOpenXMLWorkbook'))
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
    
    opts = detectImportOptions(file,'Sheet',1);
    
    if (~strcmp(opts.VariableNames(1),'Date'))
        error('The first column of the ''Returns'' table must be called ''Date'' and must contain the time series dates.');
    end
    
    opts = setvartype(opts,[{'datetime'} repmat({'double'},1,size(opts.VariableNames,2)-1)]);
    opts = setvaropts(opts,'Date','InputFormat','dd/mm/yyyy');
    rets = readtable(file,opts);

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
    
    if (frms < 5)
        error('The dataset must consider at least 5 firms in order to run consistent calculations.');
    end

    t = size(rets,1);

    if (t < 252)
        error('The dataset must consider at least 252 observations in order to run consistent calculations.');
    end
    
    opts = detectImportOptions(file,'Sheet',2);
    
    if (~strcmp(opts.VariableNames(1),'Date'))
        error('The first column of the ''Market Capitalization'' table must be called ''Date'' and must contain the time series dates.');
    end
    
    opts = setvartype(opts,[{'datetime'} repmat({'double'},1,size(opts.VariableNames,2)-1)]);
    opts = setvaropts(opts,'Date','InputFormat','dd/mm/yyyy');

    frms_cap = readtable(file,opts);
    
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

    opts = detectImportOptions(file,'Sheet',3);
    
    if (~strcmp(opts.VariableNames(1),'Date'))
        error('The first column of the ''Total Liabilities'' table must be called ''Date'' and must contain the time series dates.');
    end
    
    opts = setvartype(opts,[{'datetime'} repmat({'double'},1,size(opts.VariableNames,2)-1)]);
    opts = setvaropts(opts,'Date','InputFormat','dd/mm/yyyy');
    
    frms_lia = readtable(file,opts);
    
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
        opts = detectImportOptions(file,'Sheet',stvars_off);
        
        if (~strcmp(opts.VariableNames(1),'Date'))
            error('The first column of the ''State Variables'' table must be called ''Date'' and must contain the time series dates.');
        end
        
        opts = setvartype(opts,[{'datetime'} repmat({'double'},1,size(opts.VariableNames,2)-1)]);
        opts = setvaropts(opts,'Date','InputFormat','dd/mm/yyyy');
        
        stvars = readtable(file,opts);
        
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
        opts = detectImportOptions(file,'Sheet',grps_off);
        opts = setvartype(opts,{'double' 'char'});
        grps = readtable(file,opts);
        
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
