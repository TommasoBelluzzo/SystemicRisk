% [INPUT]
% file        = A string representing the full path to the Excel spreadsheet containing the dataset.
% date_format = A string representing the date format used in the Excel spreadsheet (optional, default=dd/MM/yyyy).
%
% [OUTPUT]
% data        = A structure containing the parsed dataset.

function data = parse_dataset(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('file',@(x)validateattributes(x,{'char'},{'nonempty','size',[1,NaN]}));
        ip.addOptional('date_format','dd/MM/yyyy',@(x)validateattributes(x,{'char'},{'nonempty','size',[1,NaN]}));
    end

    ip.parse(varargin{:});
    ipr = ip.Results;

    data = parse_dataset_internal(ipr.file,ipr.date_format);

end

function data = parse_dataset_internal(file,date_format)

    try
        datetime('now','InputFormat',date_format);
    catch
        error('The specified date format is invalid.');
    end

    if (exist(file,'file') == 0)
        error('The dataset file does not exist.');
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

    if (~strcmp(file_sheets{1},'Returns'))
        error('The first sheet of the dataset file must be the ''Returns'' one.');
    end

    file_sheets_other = file_sheets(2:end);
    capitalization_defined = any(strcmp(file_sheets_other,{'Market Capitalization'}));
    liabilities_defined = any(strcmp(file_sheets_other,{'Total Liabilities'}));
    
    if (capitalization_defined && liabilities_defined)
        full = true;
    else
        if (capitalization_defined && ~liabilities_defined)
            error('The dataset file contains the ''Market Capitalization'' sheet but not the ''Total Liabilities'' sheet.');
        elseif (~capitalization_defined && liabilities_defined)
            error('The dataset file contains the ''Total Liabilities'' sheet but not the ''Market Capitalization'' sheet.');
        end
        
        full = false;
    end

    tab_returns = parse_table(file,1,'Returns',date_format);
    
    if (any(any(ismissing(tab_returns))))
        error('The ''Returns'' table contains invalid or missing values.');
    end
    
    if (width(tab_returns) < 5)
        error('The dataset must contain at least the following series: observations dates, benchmark returns and the returns of 3 firms to analyze.');
    end
    
    t = height(tab_returns);

    if (t < 253)
        error('The dataset must contain at least 253 observations (a full business year plus an additional observation at the beginning of the time series) in order to run consistent calculations.');
    end

    dates_str = cellstr(datetime(tab_returns{:,1},'InputFormat',date_format));
    dates_num = datenum(tab_returns{:,1});
    tab_returns.Date = [];

    index_returns = tab_returns{2:end,1};
    index_name = tab_returns.Properties.VariableNames{1};
    firms = numel(tab_returns.Properties.VariableNames) - 1;
    firm_names = tab_returns.Properties.VariableNames(2:end);
    firm_returns = tab_returns{2:end,2:end};

    for tab = {'Market Capitalization' 'Total Liabilities' 'Separate Accounts' 'State Variables' 'Groups'}

        tab_index = find(strcmp(file_sheets_other,tab),1);

        switch (char(tab))

            case 'Market Capitalization'

                if (~isempty(tab_index))
                    tab_capitalizations = parse_table(file,tab_index+1,'Market Capitalization',date_format);

                    if (any(any(ismissing(tab_capitalizations))))
                        error('The ''Market Capitalization'' sheet contains invalid or missing values.');
                    end
                    
                    if (any(any(tab_capitalizations{:,2:end} < 0)))
                        error('The ''Market Capitalization'' sheet contains negative values.');
                    end

                    if ((size(tab_capitalizations,1) ~= t) || any(datenum(tab_capitalizations.Date) ~= dates_num))
                        error('The observation dates in ''Returns'' and ''Market Capitalization'' sheets are mismatching.');
                    end

                    tab_capitalizations.Date = [];

                    if (~isequal(tab_capitalizations.Properties.VariableNames,firm_names))
                        error('The firm names in ''Returns'' and ''Market Capitalization'' sheets are mismatching.');
                    end

                    capitalizations = tab_capitalizations{2:end,:};
                    capitalizations_lagged = tab_capitalizations{1:end-1,:};
                else
                    capitalizations = [];
                    capitalizations_lagged = [];
                end

            case 'Total Liabilities'
                
                if (~isempty(tab_index))
                    tab_liabilities = parse_table(file,tab_index+1,'Total Liabilities',date_format);

                    if (any(any(ismissing(tab_liabilities))))
                        error('The ''Total Liabilities'' sheet contains invalid or missing values.');
                    end
                    
                    if (any(any(tab_liabilities{:,2:end} < 0)))
                        error('The ''Total Liabilities'' sheet contains negative values.');
                    end

                    if ((size(tab_liabilities,1) ~= t) || any(datenum(tab_liabilities.Date) ~= dates_num))
                        error('The observation dates in ''Returns'' and ''Total Liabilities'' sheets are mismatching.');
                    end

                    tab_liabilities.Date = [];

                    if (~isequal(tab_liabilities.Properties.VariableNames,firm_names))
                        error('The firm names in ''Returns'' and ''Total Liabilities'' sheets are mismatching.');
                    end
                    
                    liabilities = tab_liabilities{2:end,:};
                else
                    liabilities = [];
                end
                
            case 'Separate Accounts'
                
                if (~isempty(tab_index))
                    tab_separate_accounts = parse_table(file,tab_index+1,'Separate Accounts',date_format);

                    if (any(any(ismissing(tab_separate_accounts))))
                        error('The ''Separate Accounts'' sheet contains invalid or missing values.');
                    end
                    
                    if (any(any(tab_separate_accounts{:,2:end} < 0)))
                        error('The ''Separate Accounts'' sheet contains negative values.');
                    end

                    if ((size(tab_separate_accounts,1) ~= t) || any(datenum(tab_separate_accounts.Date) ~= dates_num))
                        error('The observation dates in ''Returns'' and ''Separate Accounts'' sheets are mismatching.');
                    end

                    tab_separate_accounts.Date = [];

                    if (~isequal(tab_separate_accounts.Properties.VariableNames,firm_names))
                        error('The firm names in ''Returns'' and ''Separate Accounts'' sheets are mismatching.');
                    end
                    
                    separate_accounts = tab_separate_accounts{2:end,:};
                else
                    separate_accounts = NaN(1,t);
                end

            case 'State Variables'

                if (~isempty(tab_index))
                    tab_state_variables = parse_table(file,tab_index+1,'State Variables',date_format);

                    if (any(any(ismissing(tab_state_variables))))
                        error('The ''State Variables'' sheet contains invalid or missing values.');
                    end

                    if ((size(tab_state_variables,1) ~= t) || any(datenum(tab_state_variables.Date) ~= dates_num))
                        error('The observation dates of ''Returns'' and ''State Variables'' sheets are mismatching.');
                    end

                    tab_state_variables.Date = [];

                    state_variables = tab_state_variables{1:end-1,:};
                else
                    state_variables = [];
                end

            case 'Groups'
                
                if (~isempty(tab_index))
                    tab_groups = parse_table(file,tab_index+1,'Groups',date_format);

                    if (any(any(ismissing(tab_groups))))
                        error('The ''Groups'' sheet contains invalid or missing values.');
                    end

                    if (~isequal(tab_groups.Properties.VariableNames,{'Name' 'Count'}))
                        error('The ''Groups'' sheet contains invalid (wrong name) or misplaced (wrong order) columns.');
                    end

                    if (size(tab_groups,1) < 2)
                        error('In the ''Groups'' sheet, the number of rows must be greater than or equal to 2.');
                    end

                    groups_count = tab_groups{:,2};

                    if (any(groups_count <= 0))
                        error('The ''Groups'' sheet contains one or more groups with an invalid number of firms.');
                    end

                    if (sum(groups_count) ~= firms)
                        error('In the ''Groups'' sheet, the number of firms must be equal to the one defined in the ''Returns'' sheet.');
                    end

                    group_delimiters = cumsum(groups_count(1:end-1,:));
                    group_names = strtrim(tab_groups{:,1});
                else
                    group_delimiters = [];    
                    group_names = [];
                end

        end
    end

    data = struct();
    
    data.Full = full;
    data.T = t - 1;
    data.N = firms;
    
    data.DatesNum = dates_num(2:end);
    data.DatesStr = dates_str(2:end);
    
    data.IndexName = index_name;
    data.IndexReturns = index_returns;
    data.FirmNames = firm_names;
    data.FirmReturns = firm_returns;

    data.Capitalizations = capitalizations;
    data.CapitalizationsLagged = capitalizations_lagged;
    data.Liabilities = liabilities;
    data.SeparateAccounts = separate_accounts;
    data.StateVariables = state_variables;

    data.Groups = numel(group_names);
    data.GroupDelimiters = group_delimiters;
    data.GroupNames = group_names;

end

function output = parse_table(file,sheet,name,date_format)

    if (verLessThan('Matlab','9.1'))
        output = readtable(file,'Sheet',sheet);
        
        if (~all(cellfun(@isempty,regexp(output.Properties.VariableNames,'^Var\d+$','once'))))
            error(['The ''' name ''' table contains unnamed columns.']);
        end

        if (strcmp(name,'Groups'))
            output_vars = varfun(@class,output,'OutputFormat','cell');
            
            if (~strcmp(output_vars{1},'cell') || ~strcmp(output_vars{2},'double'))
                error(['The ''' name ''' table contains invalid or missing values.']);
            end
        else
            if (~strcmp(output.Properties.VariableNames(1),'Date'))
                error(['The first column of the ''' name ''' table must be called ''Date'' and must contain the observation dates.']);
            end
            
            output.Date = datetime(output.Date,'InputFormat',date_format);
            
            output_vars = varfun(@class,output,'OutputFormat','cell');
            
            if (~all(strcmp(output_vars(2:end),'double')))
                error(['The ''' name ''' table contains invalid or missing values.']);
            end
        end
    else
        options = detectImportOptions(file,'Sheet',sheet);
        
        if (~all(cellfun(@isempty,regexp(options.VariableNames,'^Var\d+$','once'))))
            error(['The ''' name ''' table contains unnamed columns.']);
        end

        if (strcmp(name,'Groups'))
            options = setvartype(options,{'char' 'double'});
        else
            if (~strcmp(options.VariableNames(1),'Date'))
                error(['The first column of the ''' name ''' table must be called ''Date'' and must contain the observation dates.']);
            end

            options = setvartype(options,[{'datetime'} repmat({'double'},1,numel(options.VariableNames)-1)]);
            options = setvaropts(options,'Date','InputFormat',date_format);
        end

        output = readtable(file,options);
    end

end
