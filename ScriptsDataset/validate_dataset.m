% [INPUT]
% ds = A structure representing the dataset.
% category = A string representing the type of validation to perform (optional, default='').
%
% [OUTPUT]
% ds = A structure containing the validated dataset.

function ds = validate_dataset(varargin)

    persistent categories;

    if (isempty(categories))
        categories = {
            'Comparison' 'Component' 'Connectedness' 'CrossEntropy' 'CrossQuantilogram' ...
            'CrossSectional' 'Default' 'Liquidity' 'RegimeSwitching' 'Spillover' 'TailDependence'
        };
    end

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('ds',@(x)validateattributes(x,{'struct'},{'nonempty'}));
        ip.addOptional('category','',@(x)any(validatestring(x,categories)));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    ds = ipr.ds;
    category = ipr.category;

    nargoutchk(1,1);

    ds = validate_dataset_internal(ds,category,categories);

end

function ds = validate_dataset_internal(ds,category,categories)

    validate_field(ds,'File',{'char'},{'nonempty' 'size' [1 NaN]});
    validate_field(ds,'Version',{'char'},{'nonempty' 'size' [1 NaN]});
    creation_date = validate_field(ds,'CreationDate',{'double'},{'real' 'finite' '>=' datenum(2014,1,1) 'scalar'}); %#ok<DATNM> 

    result = validate_field(ds,'Result',{'char'},{'optional' 'nonempty' 'size' [1 NaN]});
    result_date = validate_field(ds,'ResultDate',{'double'},{'optional' 'real' 'finite' '>=' datenum(2014,1,1) 'scalar'}); %#ok<DATNM> 
    result_analysis = validate_field(ds,'ResultAnalysis',{'function_handle'},{'optional' 'scalar'});
    result_serial = validate_field(ds,'ResultSerial',{'char'},{'optional' 'nonempty' 'size' [1 NaN]});

    if (isempty(result))
        if (~isempty(result_date))
            error(['The dataset field ''ResultDate'' is invalid.' new_line() 'Expected value to be empty.']);
        end

        if (~isempty(result_analysis))
            error(['The dataset field ''ResultAnalysis'' is invalid.' new_line() 'Expected value to be empty.']);
        end

        if (~isempty(result_serial))
            error(['The dataset field ''ResultSerial'' is invalid.' new_line() 'Expected value to be empty.']);
        end
    else
        if (~ismember(result,categories))
            error(['The dataset field ''Result'' is invalid.' new_line() 'Expected value to be a string containing a valid result category.']);
        end

        if (isempty(result_date) || (result_date < creation_date))
            error(['The dataset field ''ResultDate'' is invalid.' new_line() 'Expected value to be a numeric date greater than or equal to ' datestr(creation_date,'dd/mm/yyyy') '.']); %#ok<DATST> 
        end

        if (isempty(result_analysis))
            error(['The dataset field ''ResultAnalysis'' is invalid.' new_line() 'Expected value to be a function handle.']);
        end

        if (isempty(result_serial))
            error(['The dataset field ''ResultSerial'' is invalid.' new_line() 'Expected value to be a function handle.']);
        end
    end

    validate_field(ds,'TimeSeries',{'cellstr'},{'nonempty' 'size' [1 8]});

    n = validate_field(ds,'N',{'double'},{'real' 'finite' 'integer' '>=' 3 'scalar'});
    t = validate_field(ds,'T',{'double'},{'real' 'finite' 'integer' '>=' 252 'scalar'});

    validate_field(ds,'DatesNum',{'double'},{'real' 'finite' 'integer' '>' 0 'nonempty' 'size' [t 1]});
    validate_field(ds,'DatesStr',{'cellstr'},{'nonempty' 'size' [t 1]});
    validate_field(ds,'MonthlyTicks',{'logical'},{'scalar'});

    validate_field(ds,'IndexName',{'char'},{'nonempty' 'size' [1 NaN]});
    validate_field(ds,'FirmNames',{'cellstr'},{'nonempty' 'size' [1 n]});

    validate_field(ds,'Index',{'double'},{'real' 'finite' 'nonempty' 'size' [t 1]});
    validate_field(ds,'Returns',{'double'},{'real' 'nanfinite' 'nonempty' 'size' [t n]});

    validate_field(ds,'Prices',{'double'},{'optional' 'real' 'nanfinite' 'nonnegative' 'nonempty' 'size' [t n]});
    validate_field(ds,'Volumes',{'double'},{'optional' 'real' 'nanfinite' 'nonnegative' 'nonempty' 'size' [t n]});
    validate_field(ds,'Capitalizations',{'double'},{'optional' 'real' 'nanfinite' 'nonnegative' 'nonempty' 'size' [t n]});

    validate_field(ds,'RiskFreeRate',{'double'},{'optional' 'real' 'finite' 'nonempty' 'size' [t 1]});
    validate_field(ds,'CDS',{'double'},{'optional' 'real' 'nanfinite' 'nonnegative' 'nonempty' 'size' [t n]});

    validate_field(ds,'Assets',{'double'},{'optional' 'real' 'nanfinite' 'nonnegative' 'nonempty' 'size' [t n]});
    validate_field(ds,'Equity',{'double'},{'optional' 'real' 'nanfinite' 'nonempty' 'size' [t n]});
    validate_field(ds,'Liabilities',{'double'},{'optional' 'real' 'nanfinite' 'nonnegative' 'nonempty' 'size' [t n]});
    validate_field(ds,'SeparateAccounts',{'double'},{'optional' 'real' 'nanfinite' 'nonnegative' 'nonempty' 'size' [t n]});

    state_variables = validate_field(ds,'StateVariables',{'double'},{'optional' 'real' 'finite' 'nonempty' 'size' [t NaN]});
    validate_field(ds,'StateVariablesNames',{'cellstr'},{'optional' 'nonempty' 'size' [1 size(state_variables,2)]});

    groups = validate_field(ds,'Groups',{'double'},{'real' 'finite' 'integer' '>=' 0 'scalar'});

    if (groups == 0)
        validate_field(ds,'GroupDelimiters',{'double'},{'size' [0 0]});
        validate_field(ds,'GroupNames',{'double'},{'size' [0 0]});
        validate_field(ds,'GroupShortNames',{'double'},{'size' [0 0]});
    else
        validate_field(ds,'GroupDelimiters',{'double'},{'real' 'finite' 'integer' 'positive' 'increasing' 'nonempty' 'size' [(groups - 1) 1]});
        validate_field(ds,'GroupNames',{'cellstr'},{'nonempty' 'size' [groups 1]});
        validate_field(ds,'GroupShortNames',{'cellstr'},{'nonempty' 'size' [groups 1]});
    end

    crises = validate_field(ds,'Crises',{'double'},{'real' 'finite' 'integer' '>=' 0 'scalar'});
    crises_type = validate_field(ds,'CrisesType',{'char'},{'nonempty' 'size' [1 1]});

    if (~strcmp(crises_type,'E') && ~strcmp(crises_type,'R'))
        error(['The dataset field ''CrisesType'' is invalid.' new_line() 'Expected value to be either ''E'' or ''R''.']);
    end

    if (crises == 0)
        validate_field(ds,'CrisesDummy',{'double'},{'size' [0 0]});
        validate_field(ds,'CrisisDates',{'double'},{'size' [0 0]});
        validate_field(ds,'CrisisDummies',{'double'},{'size' [0 0]});
        validate_field(ds,'CrisisNames',{'double'},{'size' [0 0]});
    else
        validate_field(ds,'CrisesDummy',{'double'},{'real' 'finite' 'integer' '>=' 0 '<=' 1 'nonempty' 'size' [t 1]});

        if (strcmp(crises_type,'E'))
            validate_field(ds,'CrisisDates',{'double'},{'real' 'finite' 'integer' '>' 0 'nonempty' 'size' [crises 1]});
            validate_field(ds,'CrisisDummies',{'double'},{'size' [0 0]});
        else
            validate_field(ds,'CrisisDates',{'double'},{'real' 'finite' 'integer' '>' 0 'nonempty' 'size' [crises 2]});
            validate_field(ds,'CrisisDummies',{'double'},{'real' 'finite' 'integer' '>=' 0 '<=' 1 'nonempty' 'size' [t crises]});
        end

        validate_field(ds,'CrisisNames',{'cellstr'},{'nonempty' 'size' [crises 1]});
    end

    validate_field(ds,'Defaults',{'double'},{'real' 'nonempty' 'offset' 'size' [1 n]});
    validate_field(ds,'Insolvencies',{'double'},{'real' 'nonempty' 'offset' 'size' [1 n]});

    validate_field(ds,'SupportsComponent',{'logical'},{'scalar'});
    validate_field(ds,'SupportsConnectedness',{'logical'},{'scalar'});
    validate_field(ds,'SupportsCrossEntropy',{'logical'},{'scalar'});
    validate_field(ds,'SupportsCrossSectional',{'logical'},{'scalar'});
    validate_field(ds,'SupportsDefault',{'logical'},{'scalar'});
    validate_field(ds,'SupportsLiquidity',{'logical'},{'scalar'});
    validate_field(ds,'SupportsRegimeSwitching',{'logical'},{'scalar'});
    validate_field(ds,'SupportsSpillover',{'logical'},{'scalar'});
    validate_field(ds,'SupportsTailDependence',{'logical'},{'scalar'});
    validate_field(ds,'SupportsComparison',{'logical'},{'scalar'});

    if (~isempty(category))
        if (strcmp(category,'Comparison'))
            if (~ds.SupportsComparison)
                error('The dataset cannot be used to compare systemic risk measures.');
            end
        else
            supports = ['Supports' category];

            if (~ds.(supports))
                category_e = category;
                indices = find(isstrprop(category,'upper'));

                for i = 1:numel(indices)
                    index = indices(i);

                    if (index == 1)
                        category_e(i) = lower(category(index));
                    else
                        category_e = [category_e(1:index-1) '-' lower(category_e(index)) category_e(index+1:end)];
                        indices(i+1:end) = indices(i+1:end) + 1;
                    end
                end

                error(['The dataset cannot be used to calculate ''' category_e ''' measures.']);
            end
        end
    end

end

function value = validate_field(ds,field_name,field_type,field_validator)

    if (~isfield(ds,field_name))
        error(['The dataset does not contain the field ''' field_name '''.']);
    end

    value = ds.(field_name);
    value_iscellstr = (numel(field_type) == 1) && strcmp(field_type{1},'cellstr');
    value_isfinite = any(strcmp(field_validator,'nanfinite'));
    value_isoffset = any(strcmp(field_validator,'offset'));
    value_isoptional = strcmp(field_validator{1},'optional');

    if (value_isoptional)
        empty = false;

        try
            validateattributes(value,{'double'},{'size' [0 0]});
            empty = true;
        catch
        end

        if (empty)
            return;
        end

        field_validator = field_validator(2:end);
    end

    if (value_iscellstr)
        if (~iscellstr(value) || any(cellfun(@length,value) == 0) || any(cellfun(@(x)size(x,1),value) ~= 1)) %#ok<ISCLSTR>
            error(['The dataset field ''' field_name ''' is invalid.' new_line() 'Expected value to be a cell array of non-empty character vectors.']);
        end

        field_type{1} = 'cell';
    else
        if (value_isfinite)
            index = strcmp(field_validator,'nanfinite');
            field_validator(index) = [];
        end

        if (value_isoffset)
            index = strcmp(field_validator,'offset');
            field_validator(index) = [];
        end
    end

    try
        validateattributes(value,field_type,field_validator);
    catch e
        error(['The dataset field ''' field_name ''' is invalid.' new_line() strrep(e.message,'Expected input','Expected value')]);
    end

    if (value_isfinite)
        value_check = value;
        value_check(isnan(value_check)) = 0;

        try
            validateattributes(value_check,field_type,{'finite'});
        catch
            error(['The dataset field ''' field_name ''' is invalid.' new_line() 'Expected value to be finite.']);
        end
    end

    if (value_isoffset)
        value_check = value;
        value_check(isnan(value_check)) = 2;

        try
            validateattributes(value_check,field_type,{'>',1});
        catch
            error(['The dataset field ''' field_name ''' is invalid.' new_line() 'Expected value to contain all the finite values > 1.']);
        end
    end

end
