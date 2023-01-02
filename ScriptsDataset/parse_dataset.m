% [INPUT]
% file = A string representing the full path to the Excel spreadsheet containing the dataset.
% version = A string representing the version of the dataset.
% date_format_base = A string representing the base date format used in the Excel spreadsheet for all elements except balance sheet ones (optional, default='dd/mm/yyyy').
% date_format_balance = A string representing the date format used in the Excel spreadsheet for balance sheet elements (optional, default='QQ yyyy').
% shares_type = A string representing the type of data included in the Shares sheet (optional, default='P'):
%   - 'P' for prices;
%   - 'R' for returns.
% crises_type = A string representing the type of data included in the Crises sheet (optional, default='R'):
%   - 'E' for events;
%   - 'R' for time ranges.
% distress_threshold = A float [0.05,0.75] representing the threshold at which distress statuses are detected (optional, default=0.05).
%
% [OUTPUT]
% ds = A structure containing the parsed dataset.

function ds = parse_dataset(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('file',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addRequired('version',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addOptional('date_format_base','dd/mm/yyyy',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addOptional('date_format_balance','QQ yyyy',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addOptional('shares_type','P',@(x)any(validatestring(x,{'P' 'R'})));
        ip.addOptional('crises_type','R',@(x)any(validatestring(x,{'E' 'R'})));
        ip.addOptional('distress_threshold',0.05,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.05 '<=' 0.75 'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    file = ipr.file;
    version = validate_version(ipr.version);
    date_format_base = validate_date_format(ipr.date_format_base,true);
    date_format_balance = validate_date_format(ipr.date_format_balance,false);
    shares_type = ipr.shares_type;
    crises_type = ipr.crises_type;
    distress_threshold = ipr.distress_threshold;

    file_sheets = validate_xls(ipr.file,'D');

    nargoutchk(1,1);

    ds = parse_dataset_internal(file,file_sheets,version,date_format_base,date_format_balance,shares_type,crises_type,distress_threshold);

end

function ds = parse_dataset_internal(file,file_sheets,version,date_format_base,date_format_balance,shares_type,crises_type,distress_threshold)

    [~,file_name,file_ext] = fileparts(file);
    file_name = [file_name file_ext];

    if (~strcmp(file_sheets{1},'Shares'))
        error(['Error in dataset ''' file_name ''': the first sheet must be the ''Shares'' one.']);
    end

    file_sheets_other = file_sheets(2:end);

    using_prices = strcmp(shares_type,'P');

    if (using_prices)
        tab_shares = parse_table_standard(file,file_name,1,'Shares',date_format_base,[],[],true);
    else
        tab_shares = parse_table_standard(file,file_name,1,'Shares',date_format_base,[],[],false);
    end

    if (any(any(ismissing(tab_shares))))
        error(['Error in dataset ''' file_name ''': the ''Shares'' sheet contains invalid or missing values.']);
    end

    if (width(tab_shares) < 5)
        error(['Error in dataset ''' file_name ''': the ''Shares'' sheet must contain at least the following elements: the observations dates and the time series of the benchmark and at least 3 firms.']);
    end

    n = width(tab_shares) - 2;

    if (using_prices)
        t = height(tab_shares);

        if (t < 253)
            error(['Error in dataset ''' file_name ''': the ''Shares'' sheet must contain at least 253 observations (a full business year plus an additional observation at the beginning of the time series) in order to run consistent calculations.']);
        end

        t = t - 1;
    else
        t = height(tab_shares);

        if (t < 252)
            error(['Error in dataset ''' file_name ''': the ''Shares'' sheet must contain at least 252 observations (a full business year) in order to run consistent calculations.']);
        end
    end

    dates_num = datenum(tab_shares{:,1}); %#ok<DATNM> 
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

    if (any(cellfun(@isempty,regexpi(firm_names,'^[A-Z][A-Z0-9]{0,11}$'))))
        error(['Error in dataset ''' file_name ''': the ''Shares'' sheet contains invalid firm names (containing invalid characters, not starting with a letter and/or greater than 12 characters).']);
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

    groups = 0;
    group_delimiters = [];    
    group_names = [];
    group_short_names = [];

    crises = 0;
    crises_dummy = [];
    crisis_dates = [];
    crisis_dummies = [];
    crisis_names = [];

    for tab = {'Volumes' 'Capitalizations' 'CDS' 'Assets' 'Equity' 'Separate Accounts' 'State Variables' 'Groups' 'Crises'}

        tab_index = find(strcmp(file_sheets_other,tab),1);

        if (isempty(tab_index))
            continue;
        end

        tab_index = tab_index + 1;
        tab_name = char(tab);

        switch (tab_name)

            case 'Volumes'

                tab_volumes = parse_table_standard(file,file_name,tab_index,tab_name,date_format_base,dates_num,firm_names,true);
                volumes = table2array(tab_volumes);

            case 'Capitalizations'

                tab_capitalizations = parse_table_standard(file,file_name,tab_index,tab_name,date_format_base,dates_num,firm_names,true);
                capitalizations = table2array(tab_capitalizations);

            case 'CDS'

                tab_cds = parse_table_standard(file,file_name,tab_index,tab_name,date_format_base,dates_num,[{'RF'} firm_names],true);
                tab_arr = table2array(tab_cds);
                risk_free_rate = tab_arr(:,1);
                cds = tab_arr(:,2:end) ./ 10000;

            case 'Assets'

                tab_assets = parse_table_balance(file,file_name,tab_index,tab_name,date_format_balance,dates_num,firm_names,true);
                assets = table2array(tab_assets);

            case 'Equity'

                tab_equity = parse_table_balance(file,file_name,tab_index,tab_name,date_format_balance,dates_num,firm_names,false);
                equity = table2array(tab_equity);

            case 'Separate Accounts'

                tab_separate_accounts = parse_table_balance(file,file_name,tab_index,tab_name,date_format_balance,dates_num,firm_names,true);
                separate_accounts = table2array(tab_separate_accounts);

            case 'State Variables'

                tab_state_variables = parse_table_standard(file,file_name,tab_index,tab_name,date_format_base,dates_num,[],false);
                state_variables = table2array(tab_state_variables);
                state_variables_names = tab_state_variables.Properties.VariableNames;

             case 'Groups'

                [tab_groups,group_counts] = parse_table_groups(file,file_name,tab_index,tab_name,firm_names);
                groups = height(tab_groups);
                group_delimiters = cumsum(group_counts(1:end-1,:));
                group_names = tab_groups{:,1};
                group_short_names = tab_groups{:,2};

             case 'Crises'

                if (strcmp(crises_type,'E'))
                    [tab_crises,crises_dummy] = parse_table_crises_dates(file,file_name,tab_index,tab_name,date_format_base,dates_num);
                    crises = height(tab_crises);
                    crisis_dates = datenum(tab_crises{:,1}); %#ok<DATNM> 
                    crisis_names = tab_crises{:,2};
                else
                    [tab_crises,crisis_dummies,crises_dummy] = parse_table_crises_ranges(file,file_name,tab_index,tab_name,date_format_base,dates_num);
                    crises = height(tab_crises);
                    crisis_dates = [datenum(tab_crises{:,2}) datenum(tab_crises{:,3})]; %#ok<DATNM> 
                    crisis_names = tab_crises{:,1};
                end

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
        crises_dummy = remove_first_observation(crises_dummy);
        crisis_dummies = remove_first_observation(crisis_dummies);
    end

    [dates_year,~,~,~,~,~] = datevec(dates_num);

    if (~isempty(assets) && ~isempty(equity))
        liabilities = assets - equity;
    else
        liabilities = [];
    end

    if (strcmp(crises_type,'R') && ~isempty(crises_dummy))
        crisis_dates(:,1) = max(crisis_dates(:,1),dates_num(1));
        crisis_dates(:,2) = min(crisis_dates(:,2),dates_num(end));
    end

    if (isempty(prices))
        [defaults,insolvencies] = detect_distress(distress_threshold,returns,volumes,capitalizations,cds,equity);
    else
        [defaults,insolvencies] = detect_distress(distress_threshold,prices,volumes,capitalizations,cds,equity);
    end

    dd_limit = ceil(n * 0.25);
    defaults_cnt = sum(~isnan(defaults));
    insolvencies_cnt = sum(~isnan(insolvencies));

    if (any(defaults < 63))
        error(['Error in dataset ''' file_name ''': it contains firms that defaulted too early and must be removed.']);
    end

    if (defaults_cnt > dd_limit)
        error(['Error in dataset ''' file_name ''': it contains too many defaulted firms (' num2str(defaults_cnt) ' out of a maximum of ' num2str(dd_limit) ').']);
    end

    if (any(insolvencies < 63))
        error(['Error in dataset ''' file_name ''': it contains firms that became insolvent too early and must be removed.']);
    end

    if (insolvencies_cnt > dd_limit)
        error(['Error in dataset ''' file_name ''': it contains too many insolvent firms (' num2str(insolvencies_cnt) ' out of a maximum of ' num2str(dd_limit) ').']);
    end

    includes_volumes = ismember({'Volumes'},file_sheets_other);
    includes_capitalizations = ismember({'Capitalizations'},file_sheets_other);
    includes_cds = ismember({'CDS'},file_sheets_other);
    includes_balance_sheet = sum(ismember({'Assets' 'Equity'},file_sheets_other)) == 2;
    includes_crises = ~isempty(crises_dummy);

    if (using_prices && includes_capitalizations)
        supports_bubbles_detection = true;
    else
        supports_bubbles_detection = false;
        warning('MATLAB:SystemicRisk',['The dataset ''' file_name ''' does not meet all the requirements for the computation of bubbles detection measures (''Shares'' must be expressed as prices, the ''Capitalizations'' sheet must be included).']);
    end

    if (includes_cds)
        supports_cross_entropy = true;
    else
        supports_cross_entropy = false;
        warning('MATLAB:SystemicRisk',['The dataset ''' file_name ''' does not meet all the requirements for the computation of cross-entropy measures (the ''CDS'' sheet must be included).']);
    end

    if (includes_capitalizations && includes_balance_sheet)
        supports_cross_sectional = true;
    else
        supports_cross_sectional = false;
        warning('MATLAB:SystemicRisk',['The dataset ''' file_name ''' does not meet all the requirements for the computation of cross-sectional measures (''Assets'', ''Capitalizations'' and ''Equity'' sheets must be included).']);
    end

    if (includes_capitalizations && includes_cds && includes_balance_sheet)
        supports_default = true;
    else
        supports_default = false;
        warning('MATLAB:SystemicRisk',['The dataset ''' file_name ''' does not meet all the requirements for the computation of default measures (''Assets'', ''Capitalizations'', ''CDS'' and ''Equity'' sheets must be included).']);
    end

    if (using_prices && includes_volumes && includes_capitalizations)
        supports_liquidity = true;
    else
        supports_liquidity = false;
        warning('MATLAB:SystemicRisk',['The dataset ''' file_name ''' does not meet all the requirements for the computation of liquidity measures (''Shares'' must be expressed as prices, ''Capitalizations'' and ''Volumes'' sheets must be included).']);
    end

    if (includes_crises)
        supports_comparison = true;
    else
        supports_comparison = false;
        warning('MATLAB:SystemicRisk',['The dataset ''' file_name ''' does not meet all the requirements for the comparison of systemic risk measures (the ''Crises'' sheet must be included).']);
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

    ds.File = file;
    ds.Version = version;
    ds.CreationDate = now(); %#ok<TNOW1> 

    ds.Result = [];
    ds.ResultDate = [];
    ds.ResultAnalysis = [];
    ds.ResultSerial = [];

    ds.TimeSeries = {'Assets' 'Capitalizations' 'CDS' 'Equity' 'Liabilities' 'Returns' 'SeparateAccounts' 'Volumes'};

    ds.N = n;
    ds.T = t;

    ds.DatesNum = dates_num;
    ds.DatesStr = cellstr(datestr(datetime(dates_num,'ConvertFrom','datenum'),date_format_base)); %#ok<DATST> 
    ds.MonthlyTicks = numel(unique(dates_year)) <= 3;

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

    ds.Groups = groups;
    ds.GroupDelimiters = group_delimiters;
    ds.GroupNames = group_names;
    ds.GroupShortNames = group_short_names;

    ds.Crises = crises;
    ds.CrisesType = crises_type;
    ds.CrisesDummy = crises_dummy;
    ds.CrisisDates = crisis_dates;
    ds.CrisisDummies = crisis_dummies;
    ds.CrisisNames = crisis_names;

    ds.Defaults = defaults;
    ds.Insolvencies = insolvencies;

    ds.SupportsBubblesDetection = supports_bubbles_detection;
    ds.SupportsComponent = true;
    ds.SupportsConnectedness = true;
    ds.SupportsCrossEntropy = supports_cross_entropy;
    ds.SupportsCrossQuantilogram = true;
    ds.SupportsCrossSectional = supports_cross_sectional;
    ds.SupportsDefault = supports_default;
    ds.SupportsLiquidity = supports_liquidity;
    ds.SupportsRegimeSwitching = true;
    ds.SupportsSpillover = true;
    ds.SupportsTailDependence = true;
    ds.SupportsComparison = supports_comparison;

end

%% VALIDATION

function date_format = validate_date_format(date_format,base)

    try
        datestr(now(),date_format); %#ok<DATST,TNOW1> 
    catch e
        error(['The date format ''' date_format ''' is invalid.' new_line() strtrim(regexprep(e.message,'Format.+$',''))]);
    end

    if (~any(regexp(date_format,'(?<!y)(y{2}|y{4})(?!y)')))
        error('The date format must always include year information.');
    end

    if (any(regexp(date_format,'(?:HH|MM|SS|FFF|AM|PM)')))
        error('The date format must not include time information.');
    end

    if (base && ~any(regexp(date_format,'(?<!d)(d{2}|d{4})(?!d)')))
        error('The base date format must be tied to a daily frequency.');
    end

end

function version = validate_version(version)

    if (~any(regexp(version,'v\d+\.\d+')))
        error('The version format is invalid.');
    end

end

%% PARSING

function tab = ensure_field_consistency(file_name,name,tab,index,output_type,target_type,date_format_dt)

    switch (target_type)

        case 'char'

            if (strcmp(output_type,'cell'))
                field = vertcat(tab{:,index});

                if (~ischar(field{1}))
                    error(['Error in dataset ''' file_name ''': the ''' name ''' sheet contains invalid column types.']);
                end

                tab.(tab.Properties.VariableNames{index}) = field;
            elseif (strcmp(output_type,'char'))
                tab.(tab.Properties.VariableNames{index}) = cellstr(tab{:,index});
            else
                error(['Error in dataset ''' file_name ''': the ''' name ''' sheet contains invalid column types.']);
            end

        case 'datetime'

            if (strcmp(output_type,'cell'))
                if (iscell(tab{1,index}))
                    field = cellfun(@(x)x,tab{:,index},'UniformOutput',false);
                end

                if (ischar(field{1}))
                    field = cellfun(@(x)datetime(x,'InputFormat',date_format_dt),field,'UniformOutput',false); 
                elseif (~isdatetime(field{1}))
                    error(['Error in dataset ''' file_name ''': the ''' name ''' sheet contains invalid column types.']);
                end

                tab.(tab.Properties.VariableNames{index}) = [field{:}].';
            elseif (strcmp(output_type,'char'))
                field = cellfun(@(x)datetime(x,'InputFormat',date_format_dt),cellstr(tab{:,index}));
                tab.(tab.Properties.VariableNames{1}) = field;
            elseif (~strcmp(output_type,'datetime'))
                error(['Error in dataset ''' file_name ''': the ''' name ''' sheet contains invalid column types.']);
            end

        case 'double'

            if (strcmp(output_type,'cell'))
                if (iscell(tab{1,index}))
                    field = cellfun(@(x)x{:},tab{:,index},'UniformOutput',false);
                end

                field_first = field{1};

                if (~isa(field_first,'double') || ~isscalar(field_first))
                    error(['Error in dataset ''' file_name ''': the ''' name ''' sheet contains invalid column types.']);
                end

                tab.(tab.Properties.VariableNames{index}) = [field{:}].';
            elseif (~strcmp(output_type,'double'))
                error(['Error in dataset ''' file_name ''': the ''' name ''' sheet contains invalid column types.']);
            end

        otherwise

            error(['Invalid target type specified: ' target_type '.']);

    end

end

function tab = parse_table_balance(file,file_name,index,name,date_format,dates_num,firm_names,check_negatives)

    data_types = [{'datetime'} repmat({'double'},1,numel(firm_names))];
    tab_partial = read_table(file,file_name,index,name,date_format,data_types);

    if (~strcmp(tab_partial.Properties.VariableNames(1),'Date'))
        error(['Error in dataset ''' file_name ''': the first column of the ''' name ''' sheet must be called ''Date'' and must contain the observation dates.']);
    end

    if (any(any(~isfinite(tab_partial{:,2:end}))))
        error(['Error in dataset ''' file_name ''': the ''' name ''' sheet contains invalid values.']);
    end

    if (check_negatives && any(any(tab_partial{:,2:end} < 0)))
        error(['Error in dataset ''' file_name ''': the ''' name ''' sheet contains negative values.']);
    end

    t_current = height(tab_partial);
    dates_num_current = datenum(tab_partial.Date); %#ok<DATNM> 
    tab_partial.Date = [];

    if (t_current ~= numel(unique(dates_num_current)))
        error(['Error in dataset ''' file_name ''': the ''' name ''' sheet contains duplicate observation dates.']);
    end

    if (any(dates_num_current ~= sort(dates_num_current)))
        error(['Error in dataset ''' file_name ''': the ''' name ''' sheet contains unsorted observation dates.']);
    end

    if (t_current ~= numel(unique(cellstr(datestr(dates_num,date_format))))) %#ok<DATST> 
        error(['Error in dataset ''' file_name ''': the ''' name ''' sheet contains an invalid number of observation dates.']);
    end

    if (~isequal(tab_partial.Properties.VariableNames,firm_names))
        error(['Error in dataset ''' file_name ''': the firm names between the ''Shares'' sheet and the ''' name ''' sheet are mismatching.']);
    end

    dates_from = cellstr(datestr(dates_num_current,date_format)).'; %#ok<DATST> 
    dates_to = cellstr(datestr(dates_num,date_format)).'; %#ok<DATST> 

    if (any(~ismember(dates_to,dates_from)))
        error(['Error in dataset ''' file_name ''': the ''' name ''' sheet observation dates do not cover all the ''Shares'' sheet observation dates.']);
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

function [tab,crises_dummy] = parse_table_crises_dates(file,file_name,index,name,date_format,dates_num)

    data_types = {'datetime' 'char'};
    tab = read_table(file,file_name,index,name,date_format,data_types);

    if (~isequal(tab.Properties.VariableNames,{'Date' 'Name'}))
        error(['Error in dataset ''' file_name ''': the ''' name ''' sheet contains invalid (wrong name) or misplaced (wrong order) columns.']);
    end

    dates_num_min = min(dates_num);
    dates_num_max = max(dates_num);

    dates_num_current = datenum(tab.Date); %#ok<DATNM> 

    if (any(dates_num_current < dates_num_min) || any(dates_num_current > dates_num_max))
        error(['Error in dataset ''' file_name ''': the ''' name ''' sheet contains crisis events outside the scope of the dataset (' datestr(dates_num_min,date_format) ' - ' datestr(dates_num_max,date_format) ').']); %#ok<DATST> 
    end

    n = height(tab);
    crises_dummy = zeros(numel(dates_num),1);

    for i = 1:n
        [check,index] = ismember(datenum(tab.Date(i)),dates_num); %#ok<DATNM> 

        if (~check)
            error(['Error in dataset ''' file_name ''': the ''' name ''' sheet contains crisis events outside the scope of the dataset.']);
        end

        crises_dummy(index) = 1;
    end

    if (all(crises_dummy == 1))
        error(['Error in dataset ''' file_name ''': all the observation are considered to be part of a distress period.']);
    end

end

function [tab,crisis_dummies,crises_dummy] = parse_table_crises_ranges(file,file_name,index,name,date_format,dates_num)

    data_types = {'char' 'datetime' 'datetime'};
    tab = read_table(file,file_name,index,name,date_format,data_types);

    if (~isequal(tab.Properties.VariableNames,{'Name' 'StartDate' 'EndDate'}))
        error(['Error in dataset ''' file_name ''': the ''' name ''' sheet contains invalid (wrong name) or misplaced (wrong order) columns.']);
    end

    if (any(cellfun(@length,tab.Name) > 30))
        error(['Error in dataset ''' file_name ''': the ''' name ''' sheet contains one or more crises with a name longer than 25 characters.']);
    end

    dates_num_min = min(dates_num);
    dates_num_max = max(dates_num);

    dates_num_start = datenum(tab.StartDate); %#ok<DATNM> 
    dates_num_end = datenum(tab.EndDate); %#ok<DATNM> 

    if (any(dates_num_start > dates_num_max) || any(dates_num_end < dates_num_min))
        error(['Error in dataset ''' file_name ''': the ''' name ''' sheet contains crises outside the scope of the dataset (' datestr(dates_num_min,date_format) ' - ' datestr(dates_num_max,date_format) ').']); %#ok<DATST> 
    end

    if (any(dates_num_start > dates_num_end))
        error(['Error in dataset ''' file_name ''': the ''' name ''' sheet contains one or more crises with start date greater than end date.']);
    end

    n = height(tab);
    crisis_dummies = zeros(numel(dates_num),n);

    for i = 1:n
        seq = datenum(tab.StartDate(i):tab.EndDate(i)).'; %#ok<DATNM> 
        crisis_dummies(ismember(dates_num,seq),i) = 1;
    end

    crises_dummy = double(sum(crisis_dummies,2) > 0);

    if (all(crises_dummy == 1))
        error(['Error in dataset ''' file_name ''': all the observation are considered to be part of a distress period.']);
    end

end

function [tab,group_counts] = parse_table_groups(file,file_name,index,name,firm_names)

    data_types = {'char' 'char' 'double'};
    tab = read_table(file,file_name,index,name,[],data_types);

    if (~isequal(tab.Properties.VariableNames,{'Name' 'ShortName' 'Count'}))
        error(['Error in dataset ''' file_name ''': the ''' name ''' sheet contains invalid (wrong name) or misplaced (wrong order) columns.']);
    end

    if (any(~isfinite(tab{:,end})))
        error(['Error in dataset ''' file_name ''': the ''' name ''' sheet contains invalid values.']);
    end

    group_counts = tab.Count;

    if (any(group_counts <= 0) || any(round(group_counts) ~= group_counts))
        error(['Error in dataset ''' file_name ''': the ''' name ''' sheet contains one or more groups with an invalid number of firms.']);
    end

    if (sum(group_counts) ~= numel(firm_names))
        error(['Error in dataset ''' file_name ''': in the ''' name ''' sheet, the number of firms must be equal to the one defined in the ''Shares'' sheet.']);
    end

    tab.Name = strtrim(tab.Name);
    tab.ShortName = strtrim(tab.ShortName);

    if (any(cellfun(@isempty,regexpi(tab.ShortName,'^[A-Z][A-Z0-9]{0,4}$'))))
        error(['Error in dataset ''' file_name ''': the ''' name ''' sheet contains groups with an invalid short name (containing invalid characters, not starting with a letter and/or greater than 5 characters).']);
    end

end

function tab = parse_table_standard(file,file_name,index,name,date_format,dates_num,firm_names,check_negatives)

    if (isempty(firm_names))
        data_types = [];
    else
        data_types = [{'datetime'} repmat({'double'},1,numel(firm_names))];
    end

    tab = read_table(file,file_name,index,name,date_format,data_types);

    if (~strcmp(tab.Properties.VariableNames(1),'Date'))
        error(['Error in dataset ''' file_name ''': the first column of the ''' name ''' sheet must be called ''Date'' and must contain the observation dates.']);
    end

    if (any(any(~isfinite(tab{:,2:end}))))
        error(['Error in dataset ''' file_name ''': the ''' name ''' sheet contains invalid values.']);
    end

    if (check_negatives)
        if (~isempty(firm_names) && strcmp(firm_names{1},'RF'))
            if (any(any(tab{:,3:end} < 0)))
                error(['Error in dataset ''' file_name ''': the ''' name ''' sheet contains negative values.']);
            end
        else
            if (any(any(tab{:,2:end} < 0)))
                error(['Error in dataset ''' file_name ''': the ''' name ''' sheet contains negative values.']);
            end
        end
    end

    t_current = height(tab);
    dates_num_current = datenum(tab.Date); %#ok<DATNM> 

    if (t_current ~= numel(unique(dates_num_current)))
        error(['Error in dataset ''' file_name ''': the ''' name ''' sheet contains duplicate observation dates.']);
    end

    if (any(dates_num_current ~= sort(dates_num_current)))
        error(['Error in dataset ''' file_name ''': the ''' name ''' sheet contains unsorted observation dates.']);
    end

    if (~isempty(dates_num))
        if ((t_current ~= numel(dates_num)) || any(dates_num_current ~= dates_num))
            error(['Error in dataset ''' file_name ''': the observation dates between the ''Shares'' sheet and the ''' name ''' sheet are mismatching.']);
        end

        tab.Date = [];
    end

    if (~isempty(firm_names) && ~isequal(tab.Properties.VariableNames,firm_names))
        error(['Error in dataset ''' file_name ''': the firm names between the ''Shares'' sheet and the ''' name ''' sheet are mismatching.']);
    end

end

function tab = read_table(file,file_name,index,name,date_format,data_types)

    date_format_dt = date_format;
    date_format_dt = strrep(date_format_dt,'m','M');
    date_format_dt = strrep(date_format_dt,'QQ','QQQ');

    data_types_count = numel(data_types);

    if (verLessThan('MATLAB','9.1'))
        try
            if (ispc())
                try
                    tab = readtable(file,'Sheet',index,'Basic',true);
                catch
                    tab = readtable(file,'Sheet',index);
                end
            else
                tab = readtable(file,'Sheet',index);
            end
        catch e
            error(['Error in dataset ''' file_name ''': the ''' name ''' sheet could not be read.' new_line() e.message]);
        end

        output_vars = varfun(@class,tab,'OutputFormat','cell');

        if (data_types_count == 0)
            data_types_count = width(tab);

            if (data_types_count > 0)
                data_types = [{'datetime'} repmat({'double'},1,data_types_count - 1)];
            end
        end

        for i = 1:data_types_count
            tab = ensure_field_consistency(file_name,name,tab,i,output_vars{i},data_types{i},date_format_dt);
        end
    else
        if (ispc())
            options = detectImportOptions(file,'Sheet',index,'NumHeaderLines',0);
        else
            options = detectImportOptions(file,'Sheet',name,'NumHeaderLines',0);
        end

        if (data_types_count == 0)
            data_types_count = numel(options.VariableNames);

            if (data_types_count > 0)
                data_types = [{'datetime'} repmat({'double'},1,data_types_count - 1)];
            end
        end

        options = setvartype(options,options.VariableNames(1:data_types_count),data_types);

        dt_indices = strcmp(data_types,'datetime');

        if (any(dt_indices))
            dt_variables = options.VariableNames(dt_indices);
            options = setvaropts(options,dt_variables,'InputFormat',date_format_dt);
        end

        try
            if (ispc())
                try
                    tab = readtable(file,options,'Basic',true);
                catch
                    tab = readtable(file,options);
                end
            else
                tab = readtable(file,options);
            end
        catch e
            error(['Error in dataset ''' file_name ''': the ''' name ''' sheet could not be read.' new_line() e.message]);
        end
    end

    h = height(tab);
    w = width(tab);

    if (h == 0)
        error(['Error in dataset ''' file_name ''': the ''' name ''' sheet must contain at least 1 row.']);
    end

    imr = all(ismissing(tab),2);
    f = find(diff([false; imr; false] == 0));

    if (~isempty(f))
        indices = f(1:2:end-1);
        counts = f(2:2:end) - indices;

        index_last = indices(end);
        count_last = counts(end);

        if ((index_last + count_last - 1) == h)
            tab(index_last:end,:) = [];
        end
    end

    isc = all(ismissing(tab),1).';
    f = find(diff([false; isc; false] == 0));

    if (~isempty(f))
        indices = f(1:2:end-1);
        counts = f(2:2:end) - indices;

        index_last = indices(end);
        count_last = counts(end);

        if ((index_last + count_last - 1) == w)
            tab(:,index_last:end) = [];
        end
    end

    if ((data_types_count > 1) && (width(tab) ~= data_types_count))
        error(['Error in dataset ''' file_name ''': the ''' name ''' sheet must contain exactly ' num2str(data_types_count) ' columns.']);
    end

    if (~all(cellfun(@isempty,regexp(tab.Properties.VariableNames,'^Var\d+$','once'))))
        error(['Error in dataset ''' file_name ''': the ''' name ''' sheet contains unnamed columns.']);
    end

    if (any(any(ismissing(tab))))
        error(['Error in dataset ''' file_name ''': the ''' name ''' sheet contains missing values.']);
    end

end

%% COMPUTATIONS

function [defaults,insolvencies] = detect_distress(distress_threshold,shares,volumes,capitalizations,cds,equity)

    [t,n] = size(shares);
    limit = round(t * distress_threshold,0);

    data = {shares volumes capitalizations cds};
    k = sum(~cellfun(@isempty,data,'UniformOutput',true));

    data = [shares volumes capitalizations cds];
    defaults = NaN(1,n);

    for i = 1:n
        data_i = data(:,i:n:(n * k));
        defaults_i = NaN(1,k);

        for j = 1:k
            data_ij = data_i(:,j);

            f = find(diff([1; data_ij; 1] == 0));

            if (~isempty(f))
                indices = f(1:2:end-1);
                counts = f(2:2:end) - indices;

                index_last = indices(end);
                count_last = counts(end);

                if (((index_last + count_last - 1) == t) && (count_last >= limit))
                    defaults_i(j) = index_last;
                end
            end
        end

        defaults(i) = min(defaults_i,[],'omitnan');
    end

    insolvencies = NaN(1,n);

    if (~isempty(equity))
        for i = 1:n
            eq = equity(:,i);

            f = find(diff([false; eq < 0; false] ~= 0));

            if (~isempty(f))
                indices = f(1:2:end-1);
                counts = f(2:2:end) - indices;

                index_last = indices(end);
                count_last = counts(end);

                if (((index_last + count_last - 1) == numel(eq)) && (count_last >= limit))
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
