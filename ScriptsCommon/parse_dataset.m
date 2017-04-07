% [INPUT]
% file_src = A string representing the name of the Excel spreadsheet containing the dataset (optional, default='dataset.xlsx').
%
% [OUTPUT]
% data     = A structure containing the parsed dataset.

function data = parse_dataset(varargin)

    persistent p;

    if (isempty(p))
        p = inputParser();
        p.addOptional('file_src','dataset.xlsx',@(x)validateattributes(x,{'char'},{'nonempty','size',[1,NaN]}));
    end
    
    p.parse(varargin{:});
    res = p.Results;

    data = parse_dataset_internal(res.file_src);

end

function data = parse_dataset_internal(file_src)

    if (exist(file_src,'file') == 0)
        error('The dataset file does not exist.');
    end

    [stat,shts,fmt] = xlsfinfo(file_src);

    if (isempty(stat) || ~strcmp(fmt,'xlOpenXMLWorkbook'))
        error('The dataset file is not a valid Excel spreadsheet.');
    end
    
    shts_len = length(shts);
    
    if (shts_len < 3)
        error('The dataset does not contain all the required time series.');
    end
    
    if (shts_len > 4)
        error('The dataset contains unnecessary time series.');
    end
    
    shts_val = {'Returns' 'Market Capitalization' 'Total Liabilities' 'State Variables'};
    
    for i = 1:shts_len
        if (~strcmp(shts(i),shts_val(i)))
            error('The dataset contains invalid time series (wrong names or order).');
        end
    end

    rets = readtable(file_src,'FileType','spreadsheet','Sheet',1);
    vars = rets.Properties.VariableNames;
    
    dates_str = rets{:,1};
    dates_num = datenum(dates_str,'dd/mm/yyyy');
    frms = length(vars) - 2;
    name_idx = vars{2};
    name_frms = vars(3:end);
    ret_idx = rets{:,2};
    ret_frms = rets{:,3:end};
    t = size(rets,1);
    
    rets.Date = [];
    
    if (frms < 3)
        error('The dataset must consider at least 3 firms in order to run consistent calculations.');
    end
    
    if (t < 252)
        error('The dataset must contain at least 252 observations in order to run consistent calculations.');
    end

    mcaps = readtable(file_src,'FileType','spreadsheet','Sheet',2);
    mcaps.Date = [];
    
    if ((size(mcaps,1) - 1) ~= t)
        error('The number of observations for market capitalization does not match the number of observations for log returns plus one.');
    end

    if (~isequal(mcaps.Properties.VariableNames,name_frms))
        error('The time series of market capitalization do not match the time series of log returns.');
    end
    
    tlias = readtable(file_src,'FileType','spreadsheet','Sheet',3);
    tlias.Date = [];
    
    if (size(tlias,1) ~= t)
        error('The number of observations of total liabilities does not match the number of observations of log returns.');
    end
    
    if (~isequal(name_frms,tlias.Properties.VariableNames))
        error('The time series of total liabilities do not match the time series of log returns.');
    end

    data = struct();
    data.DatesNum = dates_num;
    data.DatesStr = dates_str;
    data.Frms = frms;
    data.MCaps = mcaps{2:end,:};
    data.MCapsLag = mcaps{1:end-1,:};
    data.NameFrms = name_frms;
    data.NameIdx = name_idx;
    data.Obs = t;
    data.RetFrms = ret_frms;
    data.RetIdx = ret_idx;
    data.TLias = tlias{:,:};

    if (shts_len == 4)
        svars = readtable(file_src,'FileType','spreadsheet','Sheet',4);
        svars.Date = [];

        svars = svars{1:end-1,:};
        
        if (size(svars,1) ~= t)
            error('The number of observations of state variables does not match the number of observations of log returns.');
        end

        data.SVars = svars; 
    else
        data.SVars = [];
    end
    
end
