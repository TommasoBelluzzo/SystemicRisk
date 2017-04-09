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
        error('The dataset does not contain all the required time series.');
    end
    
    if (shts_len > 5)
        error('The dataset contains unnecessary time series.');
    end
    
    file_shts = strtrim(file_shts);
    shts_val = {'Returns' 'Market Capitalization' 'Total Liabilities' 'State Variables' 'Groups'};
    
    for i = 1:shts_len
        if (~strcmp(file_shts(i),shts_val(i)))
            error('The dataset contains invalid time series (wrong names or order).');
        end
    end

    rets = readtable(file,'FileType','spreadsheet','Sheet',1);
    vars = strtrim(rets.Properties.VariableNames);
    
    dates_str = rets{:,1};
    dates_num = datenum(dates_str,'dd/mm/yyyy');
    frms = length(vars) - 2;
    frms_nam = vars(3:end);
    frms_ret = rets{:,3:end};
    idx_ret = rets{:,2};
    idx_nam = vars{2};
    t = size(rets,1);
    
    rets.Date = [];
    
    if (frms < 5)
        error('The dataset must consider at least 5 firms in order to run consistent calculations.');
    end
    
    if (t < 252)
        error('The dataset must contain at least 252 observations in order to run consistent calculations.');
    end

    frms_cap = readtable(file,'FileType','spreadsheet','Sheet',2);
    frms_cap.Date = [];
    
    if ((size(frms_cap,1) - 1) ~= t)
        error('The number of observations for market capitalization does not match the number of observations for log returns plus one.');
    end

    if (~isequal(strtrim(frms_cap.Properties.VariableNames),frms_nam))
        error('The time series of market capitalization do not match the time series of log returns.');
    end
    
    frms_lia = readtable(file,'FileType','spreadsheet','Sheet',3);
    frms_lia.Date = [];
    
    if (size(frms_lia,1) ~= t)
        error('The number of observations of total liabilities does not match the number of observations of log returns.');
    end
    
    if (~isequal(strtrim(frms_lia.Properties.VariableNames),frms_nam))
        error('The time series of total liabilities do not match the time series of log returns.');
    end

    if (shts_len >= 4)
        stvars = readtable(file,'FileType','spreadsheet','Sheet',4);
        stvars.Date = [];

        stvars_lag = stvars{1:end-1,:};
        
        if (size(stvars_lag,1) ~= t)
            error('The number of observations of state variables does not match the number of observations of log returns.');
        end
    else
        stvars_lag = [];
    end

    if (shts_len == 5)
        grps = readtable(file,'FileType','spreadsheet','Sheet',5);
        [grps_rows,grps_cols] = size(grps);

        if (grps_rows < 2)
            error('The number of rows in the Groups worksheet must be greater than or equal to 2.');
        end

        if (grps_cols ~= 2)
            error('The number of columns in the Groups worksheet must be equal to 2.');
        end

        if (~isequal(strtrim(grps.Properties.VariableNames),{'Delimiter' 'Name'}))
            error('The header names in the Groups worksheet are not correct.');
        end

        grps_del = grps{:,1};
        
        if (grps_del(1) < 1)
            error('The first group delimiter in the Groups worksheet must be greater than or equal to 1.');
        end
        
        if (grps_del(end-1) >= frms)
            error('The penultimate group delimiter in the Groups worksheet must less than the number of firms.');
        end
        
        if (~isnan(grps_del(end)))
            error('The last group delimiter in the Groups worksheet must be a NaN.');
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
