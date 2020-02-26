% [INPUT]
% file = A string representing the full path to the Excel spreadsheet containing the dataset.
% date_format_base = A string representing the base date format used in the Excel spreadsheet for all elements except the balance sheet ones (optional, default='dd/MM/yyyy').
% date_format_balance = A string representing the date format used in the Excel spreadsheet for balance sheet elements (optional, default='QQ yyyy').
% shares_type = A string (either 'prices' or 'returns') representing the type of data included in the Shares sheet (optional, default='prices').
% forward_rolling = An integer [3,6] representing the number of months used by the forward rolling of liabilities, which simulates the difficulty of renegotiating debt in case of financial distress (optional, default=3).
%
% [OUTPUT]
% data = A structure containing the parsed dataset.

function data = parse_dataset(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('file',@(x)validateattributes(x,{'char'},{'nonempty','size',[1,NaN]}));
        ip.addOptional('date_format_base','dd/MM/yyyy',@(x)validateattributes(x,{'char'},{'nonempty','size',[1,NaN]}));
        ip.addOptional('date_format_balance','QQ yyyy',@(x)validateattributes(x,{'char'},{'nonempty','size',[1,NaN]}));
        ip.addOptional('shares_type','prices',@(x)any(validatestring(x,{'prices','returns'})));
        ip.addOptional('forward_rolling',3,@(x)validateattributes(x,{'numeric'},{'scalar','integer','real','finite','>=',3,'<=',6}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    [file,file_sheets] = validate_file(ipr.file);
    date_format_base = validate_date_format(ipr.date_format_base,true);
    date_format_balance = validate_date_format(ipr.date_format_balance,false);
    
    nargoutchk(1,1);

    data = parse_dataset_internal(file,file_sheets,date_format_base,date_format_balance,ipr.shares_type,ipr.forward_rolling);

end

function data = parse_dataset_internal(file,file_sheets,date_format_base,date_format_balance,shares_type,forward_rolling)

    if (~strcmp(file_sheets{1},'Shares'))
        error('The first sheet of the dataset file must be the ''Shares'' one.');
    end

    file_sheets_other = file_sheets(2:end);
    file_sheets_definitions = sum(ismember({'Market Capitalization' 'Assets' 'Equity'},file_sheets_other));
    
    if (file_sheets_definitions == 3)
        supports_cross_sectional = true;
    else
        supports_cross_sectional = false;
        warning('MATLAB:SystemicRisk','The dataset file does not contain all the sheets required for the computation of cross-sectional measures (''Market Capitalization'', ''Assets'' and ''Equity'').');
    end

    if (strcmp(shares_type,'prices'))
        tab_shares = parse_table_standard(file,1,'Shares',date_format_base,[],[],true);
    else
        tab_shares = parse_table_standard(file,1,'Shares',date_format_base,[],[],false);
    end
    
    if (any(any(ismissing(tab_shares))))
        error('The ''Shares'' sheet contains invalid or missing values.');
    end
    
    if (width(tab_shares) < 5)
        error('The ''Shares'' sheet must contain at least the following elements: the observations dates, the benchmark time series and the time series of 3 firms.');
    end
    
    t = height(tab_shares);

    if (t < 253)
        error('The ''Shares'' sheet must contain at least 253 observations (a full business year plus an additional observation at the beginning of the time frame) in order to run consistent calculations.');
    end
    
    n = width(tab_shares) - 2;
    t = t - 1;

    dates_str = cellstr(datetime(tab_shares{:,1},'InputFormat',date_format_base));
    dates_num = datenum(tab_shares{:,1});
    tab_shares.Date = [];
    
    if (strcmp(shares_type,'prices'))
        tab_shares_headers = strtrim(tab_shares.Properties.VariableNames);
        tab_shares = table2array(tab_shares);
        tab_shares = diff(log(tab_shares));
        tab_shares(~isfinite(tab_shares)) = 0;
        
        index_name = tab_shares_headers{1};
        index_returns = tab_shares(:,1);
        firm_names = tab_shares_headers(2:end);
        firm_returns = tab_shares(:,2:end);
    else
        tab_shares_headers = strtrim(tab_shares.Properties.VariableNames);
        
        index_name = tab_shares_headers{1};
        index_returns = tab_shares{2:end,1};
        firm_names = tab_shares_headers(2:end);
        firm_returns = tab_shares{2:end,2:end};
    end

    capitalizations = [];
    capitalizations_lagged = [];
    assets = [];
    equity = [];
    liabilities = [];
    liabilities_rolled = [];
    separate_accounts = [];
    state_variables = [];
    group_delimiters = [];    
    group_names = [];
    
    for tab = {'Market Capitalization' 'Assets' 'Equity' 'Separate Accounts' 'State Variables' 'Groups'}

        tab_index = find(strcmp(file_sheets_other,tab),1);
        
        if (isempty(tab_index))
            continue;
        end

        tab_index = tab_index + 1;
        tab_name = char(tab);

        switch (tab_name)

            case 'Market Capitalization'
                tab_capitalizations = parse_table_standard(file,tab_index,tab_name,date_format_base,dates_num,firm_names,true);
                capitalizations = tab_capitalizations{2:end,:};
                capitalizations_lagged = tab_capitalizations{1:end-1,:};

            case 'Assets'
                tab_assets = parse_table_balance(file,tab_index,tab_name,date_format_balance,dates_num,firm_names,true);
                assets = tab_assets{2:end,:};
                
            case 'Equity'
                tab_equity = parse_table_balance(file,tab_index,tab_name,date_format_balance,dates_num,firm_names,false);
                equity = tab_equity{2:end,:};
                
            case 'Separate Accounts'
                tab_separate_accounts = parse_table_balance(file,tab_index,tab_name,date_format_balance,dates_num,firm_names,true);
                separate_accounts = tab_separate_accounts{2:end,:};

            case 'State Variables'
                tab_state_variables = parse_table_standard(file,tab_index,tab_name,date_format_base,dates_num,[],false);
                state_variables = tab_state_variables{1:end-1,:};

            case 'Groups'
                [tab_groups,groups_count] = parse_table_groups(file,tab_index,tab_name,firm_names);
                group_delimiters = cumsum(groups_count(1:end-1,:));
                group_names = strtrim(tab_groups{:,1});

        end
    end
    
    if (~isempty(assets) && ~isempty(equity))
        [liabilities,liabilities_rolled] = compute_liabilities(assets,equity,dates_num,forward_rolling);
    end
    
    [assets,equity,liabilities,liabilities_rolled,separate_accounts] = handle_defaulted_firms(firm_returns,assets,equity,liabilities,liabilities_rolled,separate_accounts);

    data = struct();

    data.N = n;
    data.T = t;
    
    data.DatesNum = dates_num(2:end);
    data.DatesStr = dates_str(2:end);
    data.MonthlyTicks = length(unique(year(data.DatesNum))) <= 3;
    
    data.IndexName = index_name;
    data.IndexReturns = index_returns;
    data.FirmNames = firm_names;
    data.FirmReturns = firm_returns;

    data.Capitalizations = capitalizations;
    data.CapitalizationsLagged = capitalizations_lagged;
    data.Assets = assets;
    data.Equity = equity;
    data.Liabilities = liabilities;
    data.LiabilitiesRolled = liabilities_rolled;
    data.SeparateAccounts = separate_accounts;
    data.StateVariables = state_variables;

    data.Groups = numel(group_names);
    data.GroupDelimiters = group_delimiters;
    data.GroupNames = group_names;
 
 	data.SupportsComponent = true;
	data.SupportsConnectedness = true;
	data.SupportsCrossSectional = supports_cross_sectional;
	data.SupportsSpillover = true;

end

%% VALIDATION

function date_format = validate_date_format(date_format,base)

    try
        datetime('now','InputFormat',date_format);
    catch
        error(['The date format ''' date_format ''' is invalid.']);
    end
    
    if (base && ~any(regexp(date_format, 'd{1,2}')))
        error('The base date format must define a daily frequency.');
    end
    
end

function [file,file_sheets] = validate_file(file)

    if (exist(file,'file') == 0)
        error('The dataset file does not exist.');
    end

    [~,~,extension] = fileparts(file);

    if (~strcmp(extension,'.xlsx'))
        error('The dataset file is not a valid Excel spreadsheet.');
    end

    if (ispc())
        [file_status,file_sheets,file_format] = xlsfinfo(file);
        
        if (isempty(file_status) || ~strcmp(file_format,'xlOpenXMLWorkbook'))
            error('The dataset file is not a valid Excel spreadsheet.');
        end
    else
        [file_status,file_sheets] = xlsfinfo(file);
        
        if (isempty(file_status))
            error('The dataset file is not a valid Excel spreadsheet.');
        end
    end

end

%% PARSING

function tab = parse_table_balance(file,sheet,name,date_format,dates_num,firm_names,check_negatives)

    if (verLessThan('MATLAB','9.1'))
        tab_partial = readtable(file,'Sheet',sheet);

        if (~all(cellfun(@isempty,regexp(tab_partial.Properties.VariableNames,'^Var\d+$','once'))))
            error(['The ''' name ''' table contains unnamed columns.']);
        end

        if (~strcmp(tab_partial.Properties.VariableNames(1),'Date'))
            error(['The first column of the ''' name ''' table must be called ''Date'' and must contain the observation dates.']);
        end

        tab_partial.Date = datetime(tab_partial.Date,'InputFormat',date_format);

        output_vars = varfun(@class,tab_partial,'OutputFormat','cell');

        if (~all(strcmp(output_vars(2:end),'double')))
            error(['The ''' name ''' table contains invalid or missing values.']);
        end
    else
        options = detectImportOptions(file,'Sheet',sheet);
        
        if (~all(cellfun(@isempty,regexp(options.VariableNames,'^Var\d+$','once'))))
            error(['The ''' name ''' table contains unnamed columns.']);
        end

        if (~strcmp(options.VariableNames(1),'Date'))
            error(['The first column of the ''' name ''' table must be called ''Date'' and must contain the observation dates.']);
        end

        options = setvartype(options,[{'datetime'} repmat({'double'},1,numel(options.VariableNames)-1)]);
        options = setvaropts(options,'Date','InputFormat',date_format);

        tab_partial = readtable(file,options);
    end
    
    if (any(any(ismissing(tab_partial))))
        error(['The ''' name ''' sheet contains invalid or missing values.']);
    end

    if (check_negatives && any(any(tab_partial{:,2:end} < 0)))
        error(['The ''' name ''' sheet contains negative values.']);
    end

    t_current = height(tab_partial);
    dates_num_current = datenum(tab_partial.Date);
    tab_partial.Date = [];
    
    if (t_current ~= numel(unique(dates_num_current)))
        error(['The ''' name ''' sheet contains duplicate observation dates.']);
    end
    
    if (any(dates_num_current ~= sort(dates_num_current)))
        error(['The ''' name ''' sheet contains unsorted observation dates.']);
    end
    
    if (~isequal(tab_partial.Properties.VariableNames,firm_names))
        error(['The firm names between the ''Shares'' sheet and the ''' name ''' sheet are mismatching.']);
    end
    
    dates_from = cellstr(datestr(dates_num_current,date_format)).';
    dates_to = cellstr(datestr(dates_num,date_format)).';
    
    if (any(~ismember(dates_to,dates_from)))
        error(['The ''' name ''' sheet observation dates do not cover all the ''Shares'' sheet observation dates.']);
    end
    
	t = numel(dates_num);
    n = numel(firm_names);
    tab = array2table(NaN(t,n),'VariableNames',tab_partial.Properties.VariableNames);

    for date = dates_from
        members_from = ismember(dates_from,date);
        members_to = ismember(dates_to,date);
        tab{members_to,:} = repmat(tab_partial{members_from,:},sum(members_to),1);
    end

end

function [tab,groups_count] = parse_table_groups(file,sheet,name,firm_names)

    if (verLessThan('MATLAB','9.1'))
        tab = readtable(file,'Sheet',sheet);
        
        if (~all(cellfun(@isempty,regexp(tab.Properties.VariableNames,'^Var\d+$','once'))))
            error(['The ''' name ''' table contains unnamed columns.']);
        end

        output_vars = varfun(@class,tab,'OutputFormat','cell');

        if (~strcmp(output_vars{1},'cell') || ~strcmp(output_vars{2},'double'))
            error(['The ''' name ''' table contains invalid or missing values.']);
        end
    else
        options = detectImportOptions(file,'Sheet',sheet);
        
        if (~all(cellfun(@isempty,regexp(options.VariableNames,'^Var\d+$','once'))))
            error(['The ''' name ''' table contains unnamed columns.']);
        end

        options = setvartype(options,{'char' 'double'});
        tab = readtable(file,options);
    end
    
    if (any(any(ismissing(tab))))
        error(['The ''' name ''' sheet contains invalid or missing values.']);
    end

    if (~isequal(tab.Properties.VariableNames,{'Name' 'Count'}))
        error(['The ''' name ''' sheet contains invalid (wrong name) or misplaced (wrong order) columns.']);
    end

    if (size(tab,1) < 2)
        error(['In the ''' name ''' sheet, the number of rows must be greater than or equal to 2.']);
    end

    groups_count = tab{:,2};

    if (any(groups_count <= 0) || any(round(groups_count) ~= groups_count))
        error(['The ''' name ''' sheet contains one or more groups with an invalid number of firms.']);
    end

    if (sum(groups_count) ~= numel(firm_names))
        error(['In the ''' name ''' sheet, the number of firms must be equal to the one defined in the ''Shares'' sheet.']);
    end

end

function tab = parse_table_standard(file,sheet,name,date_format,dates_num,firm_names,check_negatives)

    if (verLessThan('MATLAB','9.1'))
        tab = readtable(file,'Sheet',sheet);
        
        if (~all(cellfun(@isempty,regexp(tab.Properties.VariableNames,'^Var\d+$','once'))))
            error(['The ''' name ''' table contains unnamed columns.']);
        end

        if (~strcmp(tab.Properties.VariableNames(1),'Date'))
            error(['The first column of the ''' name ''' table must be called ''Date'' and must contain the observation dates.']);
        end

        tab.Date = datetime(tab.Date,'InputFormat',date_format);

        output_vars = varfun(@class,tab,'OutputFormat','cell');

        if (~all(strcmp(output_vars(2:end),'double')))
            error(['The ''' name ''' table contains invalid or missing values.']);
        end
    else
        options = detectImportOptions(file,'Sheet',sheet);
        
        if (~all(cellfun(@isempty,regexp(options.VariableNames,'^Var\d+$','once'))))
            error(['The ''' name ''' table contains unnamed columns.']);
        end

        if (~strcmp(options.VariableNames(1),'Date'))
            error(['The first column of the ''' name ''' table must be called ''Date'' and must contain the observation dates.']);
        end

        options = setvartype(options,[{'datetime'} repmat({'double'},1,numel(options.VariableNames)-1)]);
        options = setvaropts(options,'Date','InputFormat',date_format);

        tab = readtable(file,options);
    end
    
    if (any(any(ismissing(tab))))
        error(['The ''' name ''' sheet contains invalid or missing values.']);
    end

    if (check_negatives && any(any(tab{:,2:end} < 0)))
        error(['The ''' name ''' sheet contains negative values.']);
    end
    
    t_current = height(tab);
    dates_num_current = datenum(tab.Date);
    
    if (t_current ~= numel(unique(dates_num_current)))
        error(['The ''' name ''' sheet contains duplicate observation dates.']);
    end
    
    if (any(dates_num_current ~= sort(dates_num_current)))
        error(['The ''' name ''' sheet contains unsorted observation dates.']);
    end
    
    if (~isempty(dates_num))
        if ((t_current ~= numel(dates_num)) || any(dates_num_current ~= dates_num))
            error(['The observation dates between the ''Shares'' sheet and the ''' name ''' sheet are mismatching.']);
        end
        
        tab.Date = [];
    end

    if (~isempty(firm_names) && ~isequal(tab.Properties.VariableNames,firm_names))
        error(['The firm names between the ''Shares'' sheet and the ''' name ''' sheet are mismatching.']);
    end

end

%% COMPUTATIONS

function [liabilities,liabilities_rolled] = compute_liabilities(assets,equity,dates_num,forward_rolling)

    liabilities = assets - equity;
    liabilities_unique = unique(liabilities,'rows','stable');

    dates_num = dates_num(2:end);
    [~,a] = unique(cellstr(datestr(dates_num,'mm/yyyy')),'stable');

    seq = a(1:forward_rolling:numel(a)) - 1;
    seq(end+1) = numel(dates_num);

    liabilities_rolled = NaN(size(liabilities));

    for i = 2:numel(seq)
        indices = (seq(i-1) + 1):seq(i);
        liabilities_rolled(indices,:) = repmat(liabilities_unique(i-1,:),numel(indices),1);
    end

end

function [assets,equity,liabilities,liabilities_rolled,separate_accounts] = handle_defaulted_firms(firm_returns,assets,equity,liabilities,liabilities_rolled,separate_accounts)

    n = size(firm_returns,2);
    t = size(firm_returns,1);

    for i = 1:n
        x = firm_returns(:,i);

        f = find(diff([1; x; 1] == 0));
        indices = f(1:2:end-1);
        counts = f(2:2:end) - indices;

        index_last = indices(end);
        count_last = counts(end);

        if ((index_last + count_last - 1) == t)
            if (~isempty(assets))
                assets(index_last:end,i) = 0;
            end

            if (~isempty(equity))
                equity(index_last:end,i) = 0;
            end

            if (~isempty(liabilities))
                liabilities(index_last:end,i) = 0;
            end

            if (~isempty(liabilities_rolled))
                liabilities_rolled(index_last:end,i) = 0;
            end

            if (~isempty(separate_accounts))
                separate_accounts(index_last:end,i) = 0; 
            end
        end
    end

end
