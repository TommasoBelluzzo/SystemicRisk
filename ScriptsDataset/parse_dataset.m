% [INPUT]
% file = A string representing the full path to the Excel spreadsheet containing the dataset.
% version = A string representing the version of the dataset.
% date_format_base = A string representing the base date format used in the Excel spreadsheet for all elements except balance sheet ones (optional, default='dd/mm/yyyy').
% date_format_balance = A string representing the date format used in the Excel spreadsheet for balance sheet elements (optional, default='QQ yyyy').
% shares_type = A string (either 'P' for prices or 'R' for returns) representing the type of data included in the Shares sheet (optional, default='P').
%
% [OUTPUT]
% ds = A structure containing the parsed dataset.

function ds = parse_dataset(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('file',@(x)validateattributes(x,{'char'},{'nonempty','size',[1,NaN]}));
        ip.addRequired('version',@(x)validateattributes(x,{'char'},{'nonempty','size',[1,NaN]}));
        ip.addOptional('date_format_base','dd/mm/yyyy',@(x)validateattributes(x,{'char'},{'nonempty','size',[1,NaN]}));
        ip.addOptional('date_format_balance','QQ yyyy',@(x)validateattributes(x,{'char'},{'nonempty','size',[1,NaN]}));
        ip.addOptional('shares_type','P',@(x)any(validatestring(x,{'P','R'})));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    [file,file_sheets] = validate_file(ipr.file);
    version = validate_version(ipr.version);
    date_format_base = validate_date_format(ipr.date_format_base,true);
    date_format_balance = validate_date_format(ipr.date_format_balance,false);
    
    nargoutchk(1,1);

    ds = parse_dataset_internal(file,file_sheets,version,date_format_base,date_format_balance,ipr.shares_type);

end

function ds = parse_dataset_internal(file,file_sheets,version,date_format_base,date_format_balance,shares_type)

    if (~strcmp(file_sheets{1},'Shares'))
        error('The first sheet of the dataset file must be the ''Shares'' one.');
    end
    
    using_prices = strcmp(shares_type,'P');

    file_sheets_other = file_sheets(2:end);
    includes_volumes = ismember({'Volumes'},file_sheets_other);
    includes_capitalizations = ismember({'Capitalizations'},file_sheets_other);
    includes_cds = ismember({'CDS'},file_sheets_other);
    includes_balance_sheet = sum(ismember({'Assets' 'Equity'},file_sheets_other)) == 2;

    if (includes_capitalizations && includes_balance_sheet)
        supports_cross_sectional = true;
    else
        supports_cross_sectional = false;
        warning('MATLAB:SystemicRisk','The dataset file does not contain all the sheets required for the computation of cross-sectional measures (''Assets'', ''Capitalizations'' and ''Equity'').');
    end
    
    if (using_prices && includes_volumes && includes_capitalizations)
        supports_liquidity = true;
    else
        supports_liquidity = false;
        warning('MATLAB:SystemicRisk','The dataset file does not meet all the requirements for the computation of liquidity measures (''Shares'' must be expressed as prices, ''Capitalizations'' and ''Volumes'' sheets must be included).');
    end
    
    if (includes_capitalizations && includes_cds && includes_balance_sheet)
        supports_default = true;
    else
        supports_default = false;
        warning('MATLAB:SystemicRisk','The dataset file does not contain all the sheets required for the computation of default measures (''Assets'', ''Capitalizations'', ''CDS'' and ''Equity'').');
    end

    if (using_prices)
        tab_shares = parse_table_standard(file,1,'Shares',date_format_base,[],[],true);
    else
        tab_shares = parse_table_standard(file,1,'Shares',date_format_base,[],[],false);
    end
    
    if (any(any(ismissing(tab_shares))))
        error('The ''Shares'' sheet contains invalid or missing values.');
    end
    
    if (width(tab_shares) < 5)
        error('The ''Shares'' sheet must contain at least the following elements: the observations dates and the time series of the benchmark and at least 3 firms.');
    end
    
    n = width(tab_shares) - 2;
    
    if (using_prices)
        t = height(tab_shares);
        
        if (t < 253)
            error('The ''Shares'' sheet must contain at least 253 observations (a full business year plus an additional observation at the beginning of the time series) in order to run consistent calculations.');
        end
        
        t = t - 1;
    else
        t = height(tab_shares);
        
        if (t < 252)
            error('The ''Shares'' sheet must contain at least 252 observations (a full business year) in order to run consistent calculations.');
        end
    end

    dates_num = datenum(tab_shares{:,1});
    tab_shares.Date = [];
    
    if (using_prices)
        tab_shares_headers = strtrim(tab_shares.Properties.VariableNames);

        prices = table2array(tab_shares);
        dlp = diff(log(prices));
        dlp(~isfinite(dlp)) = 0;
        prices = prices(:,2:end);
        
        index_name = tab_shares_headers{1};
        firm_names = tab_shares_headers(2:end);

        index = dlp(:,1);
        returns = dlp(:,2:end);
    else
        tab_shares_headers = strtrim(tab_shares.Properties.VariableNames);
        
        prices = [];
        
        index_name = tab_shares_headers{1};
        firm_names = tab_shares_headers(2:end);
        
        index = tab_shares{:,1};
        returns = tab_shares{:,2:end};
    end

    volumes = [];
    capitalizations = [];

    risk_free_rate = [];
    cds = [];

    assets = [];
    equity = [];
    separate_accounts = [];

    state_variables = [];
    state_variables_names = [];

    group_delimiters = [];    
    group_names = [];

    for tab = {'Volumes' 'Capitalizations' 'CDS' 'Assets' 'Equity' 'Separate Accounts' 'State Variables' 'Groups'}

        tab_index = find(strcmp(file_sheets_other,tab),1);
        
        if (isempty(tab_index))
            continue;
        end

        tab_index = tab_index + 1;
        tab_name = char(tab);

        switch (tab_name)
            
            case 'Volumes'
                tab_volumes = parse_table_standard(file,tab_index,tab_name,date_format_base,dates_num,firm_names,true);
                volumes = table2array(tab_volumes);

            case 'Capitalizations'
                tab_capitalizations = parse_table_standard(file,tab_index,tab_name,date_format_base,dates_num,firm_names,true);
                capitalizations = table2array(tab_capitalizations);

            case 'CDS'
                tab_cds = parse_table_standard(file,tab_index,tab_name,date_format_base,dates_num,[{'RF'} firm_names],true);
                tab_arr = table2array(tab_cds);
                risk_free_rate = tab_arr(:,1);
                cds = tab_arr(:,2:end) ./ 10000;
                
            case 'Assets'
                tab_assets = parse_table_balance(file,tab_index,tab_name,date_format_balance,dates_num,firm_names,true);
                assets = table2array(tab_assets);

            case 'Equity'
                tab_equity = parse_table_balance(file,tab_index,tab_name,date_format_balance,dates_num,firm_names,false);
                equity = table2array(tab_equity);

            case 'Separate Accounts'
                tab_separate_accounts = parse_table_balance(file,tab_index,tab_name,date_format_balance,dates_num,firm_names,true);
                separate_accounts = table2array(tab_separate_accounts);

            case 'State Variables'
                tab_state_variables = parse_table_standard(file,tab_index,tab_name,date_format_base,dates_num,[],false);
                state_variables = table2array(tab_state_variables);
                state_variables_names = tab_state_variables.Properties.VariableNames;

             case 'Groups'
                [tab_groups,groups_count] = parse_table_groups(file,tab_index,tab_name,firm_names);
                group_delimiters = cumsum(groups_count(1:end-1,:));
                group_names = strtrim(tab_groups{:,1});

        end
    end
    
    if (using_prices)
        dates_num = remove_first_observation(dates_num);
        prices = remove_first_observation(prices);
        volumes = remove_first_observation(volumes);
        capitalizations = remove_first_observation(capitalizations);
        risk_free_rate = remove_first_observation(risk_free_rate);
        cds = remove_first_observation(cds);
        assets = remove_first_observation(assets);
        equity = remove_first_observation(equity);
        separate_accounts = remove_first_observation(separate_accounts);
        state_variables = remove_first_observation(state_variables);
    end

    if (~isempty(assets) && ~isempty(equity))
        liabilities = assets - equity;
    else
        liabilities = [];
    end
    
    [defaults,insolvencies] = detect_distress(returns,equity);
    
    if (any(defaults == 1))
        error('The dataset contains firms defaulted since the beginning of the observation period that must be removed.');
    end
    
    if (sum(isnan(defaults)) < 3)
        error('The dataset contains observations in which less than 3 firms are not defaulted.');
    end
    
    if (any(insolvencies == 1))
        error('The dataset contains firms being insolvent since the beginning of the observation period that must be removed.');
    end

    if (sum(isnan(defaults)) < 3)
        error('The dataset contains observations in which less than 3 firms are not insolvent.');
    end
    
    returns = distress_data(returns,defaults);
    prices = distress_data(prices,defaults);
    volumes = distress_data(volumes,defaults);
    capitalizations = distress_data(capitalizations,defaults);
    cds = distress_data(cds,defaults);
    assets = distress_data(assets,defaults);
    equity = distress_data(equity,defaults);
    liabilities = distress_data(liabilities,defaults);
    separate_accounts = distress_data(separate_accounts,defaults);

    ds = struct();

    ds.TimeSeries = {'Assets' 'Capitalizations' 'CDS' 'Equity' 'Liabilities' 'Returns' 'SeparateAccounts' 'Volumes'};
 
    ds.File = file;
    ds.Version = version;
    
    ds.N = n;
    ds.T = t;

    ds.DatesNum = dates_num;
    ds.DatesStr = cellstr(datetime(dates_num,'ConvertFrom','datenum'));
    ds.MonthlyTicks = length(unique(year(dates_num))) <= 3;

    ds.IndexName = index_name;
    ds.FirmNames = firm_names;
    
    ds.Index = index;
    ds.Returns = returns;
    
    ds.Prices = prices;
    ds.Volumes = volumes;
    ds.Capitalizations = capitalizations;

    ds.RiskFreeRate = risk_free_rate;
    ds.CDS = cds;

    ds.Assets = assets;
    ds.Equity = equity;
    ds.Liabilities = liabilities;
    ds.SeparateAccounts = separate_accounts;

    ds.StateVariables = state_variables;
    ds.StateVariablesNames = state_variables_names;

    ds.Groups = numel(group_names);
    ds.GroupDelimiters = group_delimiters;
    ds.GroupNames = group_names;
    
    ds.Defaults = defaults;
    ds.Insolvencies = insolvencies;
    
 	ds.SupportsComponent = true;
	ds.SupportsConnectedness = true;
 	ds.SupportsCrossQuantilogram = true;
	ds.SupportsCrossSectional = supports_cross_sectional;
	ds.SupportsDefault = supports_default;
    ds.SupportsLiquidity = supports_liquidity;
	ds.SupportsSpillover = true;

end

%% VALIDATION

function date_format = validate_date_format(date_format,base)

    try
        datestr(now(),date_format);
    catch e
        error(['The date format ''' date_format ''' is invalid.' newline() strtrim(regexprep(e.message,'Format.+$',''))]);
    end
    
    if (base && ~any(regexp(date_format,'d{1,4}')))
        error('The base date format must define a daily frequency.');
    end
    
    if (any(regexp(date_format,'(?:HH|MM|SS|FFF|AM|PM)')))
        error('The date format must not include time information.');
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

    if (verLessThan('MATLAB','9.7'))
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
    else
        try
            file_sheets = sheetnames(file);
        catch
            error('The dataset file is not a valid Excel spreadsheet.');
        end
    end

end

function version = validate_version(version)

    if (~any(regexp(version,'v\d+\.\d+')))
        error('The version format is invalid.');
    end
    
end

%% PARSING

function tab = parse_table_balance(file,index,name,date_format,dates_num,firm_names,check_negatives)

    if (verLessThan('MATLAB','9.1'))
        if (ispc())
            try
            	tab_partial = readtable(file,'Sheet',index,'Basic',true);
            catch
                tab_partial = readtable(file,'Sheet',index);
            end
        else
            tab_partial = readtable(file,'Sheet',index);
        end

        if (~all(cellfun(@isempty,regexp(tab_partial.Properties.VariableNames,'^Var\d+$','once'))))
            error(['The ''' name ''' table contains unnamed columns.']);
        end

        if (~strcmp(tab_partial.Properties.VariableNames(1),'Date'))
            error(['The first column of the ''' name ''' table must be called ''Date'' and must contain the observation dates.']);
        end

        tab_partial.Date = datetime(tab_partial.Date,'InputFormat',strrep(date_format,'m','M'));

        output_vars = varfun(@class,tab_partial,'OutputFormat','cell');

        if (~all(strcmp(output_vars(2:end),'double')))
            error(['The ''' name ''' table contains invalid or missing values.']);
        end
    else
        if (ispc())
            options = detectImportOptions(file,'Sheet',index);
        else
            options = detectImportOptions(file,'Sheet',name);
        end
        
        if (~all(cellfun(@isempty,regexp(options.VariableNames,'^Var\d+$','once'))))
            error(['The ''' name ''' table contains unnamed columns.']);
        end

        if (~strcmp(options.VariableNames(1),'Date'))
            error(['The first column of the ''' name ''' table must be called ''Date'' and must contain the observation dates.']);
        end

        options = setvartype(options,[{'datetime'} repmat({'double'},1,numel(options.VariableNames)-1)]);
        options = setvaropts(options,'Date','InputFormat',strrep(date_format,'m','M'));

        if (ispc())
            try
            	tab_partial = readtable(file,options,'Basic',true);
            catch
                tab_partial = readtable(file,options);
            end
        else
            tab_partial = readtable(file,options);
        end
    end
    
    if (any(any(ismissing(tab_partial))) || any(any(~isfinite(tab_partial{:,2:end}))))
        error(['The ''' name ''' sheet contains invalid or missing values.']);
    end

    if (check_negatives && any(any(tab_partial{:,2:end} < 0)))
        error(['The ''' name ''' sheet contains negative values.']);
    end

    t_current = height(tab_partial);
    dates_num_current = datenum(tab_partial.Date);
    tab_partial.Date = [];

    if (t_current ~= numel(unique(cellstr(datestr(dates_num,date_format)))))
        error(['The ''' name ''' sheet contains an invalid number of observation dates.']);
    end
    
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

function [tab,groups_count] = parse_table_groups(file,index,name,firm_names)

    if (verLessThan('MATLAB','9.1'))
        if (ispc())
            try
            	tab = readtable(file,'Sheet',index,'Basic',true);
            catch
                tab = readtable(file,'Sheet',index);
            end
        else
            tab = readtable(file,'Sheet',index);
        end
        
        if (~all(cellfun(@isempty,regexp(tab.Properties.VariableNames,'^Var\d+$','once'))))
            error(['The ''' name ''' table contains unnamed columns.']);
        end

        output_vars = varfun(@class,tab,'OutputFormat','cell');

        if (~strcmp(output_vars{1},'cell') || ~strcmp(output_vars{2},'double'))
            error(['The ''' name ''' table contains invalid or missing values.']);
        end
    else
        if (ispc())
            options = detectImportOptions(file,'Sheet',index);
        else
            options = detectImportOptions(file,'Sheet',name);
        end
        
        if (~all(cellfun(@isempty,regexp(options.VariableNames,'^Var\d+$','once'))))
            error(['The ''' name ''' table contains unnamed columns.']);
        end

        options = setvartype(options,{'char' 'double'});
        
        if (ispc())
            try
            	tab = readtable(file,options,'Basic',true);
            catch
                tab = readtable(file,options);
            end
        else
            tab = readtable(file,options);
        end
    end
    
    if (any(any(ismissing(tab))) || any(any(~isfinite(tab{:,2:end}))))
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

function tab = parse_table_standard(file,index,name,date_format,dates_num,firm_names,check_negatives)

    if (verLessThan('MATLAB','9.1'))
        if (ispc())
            try
            	tab = readtable(file,'Sheet',index,'Basic',true);
            catch
                tab = readtable(file,'Sheet',index);
            end
        else
            tab = readtable(file,'Sheet',index);
        end
        
        if (~all(cellfun(@isempty,regexp(tab.Properties.VariableNames,'^Var\d+$','once'))))
            error(['The ''' name ''' table contains unnamed columns.']);
        end

        if (~strcmp(tab.Properties.VariableNames(1),'Date'))
            error(['The first column of the ''' name ''' table must be called ''Date'' and must contain the observation dates.']);
        end

        tab.Date = datetime(tab.Date,'InputFormat',strrep(date_format,'m','M'));

        output_vars = varfun(@class,tab,'OutputFormat','cell');

        if (~all(strcmp(output_vars(2:end),'double')))
            error(['The ''' name ''' table contains invalid or missing values.']);
        end
    else
        if (ispc())
            options = detectImportOptions(file,'Sheet',index);
        else
            options = detectImportOptions(file,'Sheet',name);
        end
        
        if (~all(cellfun(@isempty,regexp(options.VariableNames,'^Var\d+$','once'))))
            error(['The ''' name ''' table contains unnamed columns.']);
        end

        if (~strcmp(options.VariableNames(1),'Date'))
            error(['The first column of the ''' name ''' table must be called ''Date'' and must contain the observation dates.']);
        end

        options = setvartype(options,[{'datetime'} repmat({'double'},1,numel(options.VariableNames) - 1)]);
        options = setvaropts(options,'Date','InputFormat',strrep(date_format,'m','M'));

        if (ispc())
            try
            	tab = readtable(file,options,'Basic',true);
            catch
                tab = readtable(file,options);
            end
        else
            tab = readtable(file,options);
        end
    end
    
    if (any(any(ismissing(tab))) || any(any(~isfinite(tab{:,2:end}))))
        error(['The ''' name ''' sheet contains invalid or missing values.']);
    end
    
    if (check_negatives)
        if (~isempty(firm_names) && strcmp(firm_names{1},'RF'))
            if (any(any(tab{:,3:end} < 0)))
                error(['The ''' name ''' sheet contains negative values.']);
            end
        else
            if (any(any(tab{:,2:end} < 0)))
                error(['The ''' name ''' sheet contains negative values.']);
            end
        end
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

function [defaults,insolvencies] = detect_distress(returns,equity)

    n = size(returns,2);
    t = size(returns,1);
    threshold = round(t * 0.05,0);
    
    defaults = NaN(1,n);

    for i = 1:n
        r = returns(:,i);

        f = find(diff([1; r; 1] == 0));
        indices = f(1:2:end-1);
        counts = f(2:2:end) - indices;

        index_last = indices(end);
        count_last = counts(end);

        if (((index_last + count_last - 1) == t) && (count_last >= threshold))
            defaults(i) = index_last;
        end
    end
    
    insolvencies = NaN(1,n);
    
    if (~isempty(equity))
        for i = 1:n
            eq = equity(:,i);

            f = find(diff([false; eq < 0; false] ~= 0));
            indices = f(1:2:end-1);

            if (~isempty(indices))
                counts = f(2:2:end) - indices;

                index_last = indices(end);
                count_last = counts(end);

                if (((index_last + count_last - 1) == numel(eq)) && (count_last >= threshold))
                    insolvencies(i) = index_last;
                end
            end   
        end
    end

end

function data = remove_first_observation(data)

    if (~isempty(data))
        data = data(2:end,:);
    end

end

