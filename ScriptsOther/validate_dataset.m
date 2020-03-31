% [INPUT]
% data = A structure representing the dataset.
% measures = A string (one of 'component', 'connectedness', 'cross-sectional', 'spillover') representing the category of measures being calculated (optional, default='').

function data = validate_dataset(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data',@(x)validateattributes(x,{'struct'},{'nonempty'}));
        ip.addOptional('measures','',@(x)any(validatestring(x,{'component','connectedness','cross-sectional','default','spillover'})));
    end

    ip.parse(varargin{:});
    ipr = ip.Results;
    
    nargoutchk(1,1);

    data = validate_dataset_internal(ipr.data,ipr.measures);

end

function data = validate_dataset_internal(data,measures)

    validate_field(data,'BinaryVersion',{'double'},{'real','finite','scalar','>=',1});
    validate_field(data,'TimeSeries',{'cellstr'},{'nonempty','size',[1 9]});

    n = validate_field(data,'N',{'numeric'},{'scalar','integer','real','finite','>=',3});
    t = validate_field(data,'T',{'numeric'},{'scalar','integer','real','finite','>=',252});

    validate_field(data,'DatesNum',{'numeric'},{'integer','real','finite','>',0,'nonempty','size',[t 1]});
    validate_field(data,'DatesStr',{'cellstr'},{'nonempty','size',[t 1]});
    validate_field(data,'MonthlyTicks',{'logical'},{'scalar'});

    validate_field(data,'IndexName',{'char'},{'nonempty','size',[1 NaN]});
    validate_field(data,'FirmNames',{'cellstr'},{'nonempty','size',[1 n]});
    
    validate_field(data,'Index',{'double'},{'real','finite','nonempty','size',[t 1]});
    validate_field(data,'Returns',{'double'},{'real','nanfinite','nonempty','size',[t n]});

    validate_field(data,'Capitalization',{'double'},{'optional','real','nanfinite','nonnegative','nonempty','size',[t n]});
    validate_field(data,'CapitalizationLagged',{'double'},{'optional','real','nanfinite','nonnegative','nonempty','size',[t n]});

    validate_field(data,'RiskFreeRate',{'double'},{'optional','real','finite','nonempty','size',[t 1]});
    validate_field(data,'CDS',{'double'},{'optional','real','nanfinite','nonnegative','nonempty','size',[t n]});

    validate_field(data,'Assets',{'double'},{'optional','real','nanfinite','nonnegative','nonempty','size',[t n]});
    validate_field(data,'Equity',{'double'},{'optional','real','nanfinite','nonempty','size',[t n]});
    validate_field(data,'Liabilities',{'double'},{'optional','real','nanfinite','nonnegative','nonempty','size',[t n]});
    validate_field(data,'LiabilitiesRolled',{'double'},{'optional','real','nanfinite','nonnegative','nonempty','size',[t n]});
    validate_field(data,'SeparateAccounts',{'double'},{'optional','real','nanfinite','nonnegative','nonempty','size',[t n]});

    state_variables = validate_field(data,'StateVariables',{'double'},{'optional','real','finite','nonempty','size',[t NaN]});
    validate_field(data,'StateVariablesNames',{'cellstr'},{'optional','nonempty','size',[1 size(state_variables,2)]});

    groups = validate_field(data,'Groups',{'numeric'},{'scalar','integer','real','finite','>=',0});

    if (groups == 0)
        validate_field(data,'GroupDelimiters',{'numeric'},{'size',[0,0]});
    else
        validate_field(data,'GroupDelimiters',{'numeric'},{'integer','real','finite','positive','increasing','nonempty','size',[(groups - 1) 1]});
    end

    if (groups == 0)
        validate_field(data,'GroupNames',{'numeric'},{'size',[0 0]});
    else
        validate_field(data,'GroupNames',{'cellstr'},{'nonempty','size',[groups 1]});
    end
    
    validate_field(data,'Defaults',{'double'},{'real','nonempty','offset','size',[1 n]});
    validate_field(data,'Insolvencies',{'double'},{'real','nonempty','offset','size',[1 n]});

    validate_field(data,'SupportsComponent',{'logical'},{'scalar'});
    validate_field(data,'SupportsConnectedness',{'logical'},{'scalar'});
    validate_field(data,'SupportsCrossSectional',{'logical'},{'scalar'});
    validate_field(data,'SupportsDefault',{'logical'},{'scalar'});
    validate_field(data,'SupportsSpillover',{'logical'},{'scalar'});
    
    if (~isempty(measures))
        measuresfinal = [upper(measures(1)) measures(2:end)];
        measures_underscore = strfind(measuresfinal,'-');

        if (~isempty(measures_underscore))
            measuresfinal(measures_underscore) = [];
            measuresfinal(measures_underscore) = upper(measuresfinal(measures_underscore));
        end

        supports = ['Supports' measuresfinal];

        if (~data.(supports))
            error(['The dataset does not contain all the required data for calculating ''' measures ''' measures.']);
        end
    end
    
end

function value = validate_field(data,field_name,field_type,field_validator)

    if (~isfield(data,field_name))
        error(['The dataset does not contain the field ''' field_name '''.']);
    end

    value = data.(field_name);
    value_iscell = (numel(field_type) == 1) && strcmp(field_type{1},'cellstr');
    value_isfinite = any(strcmp(field_validator,'nanfinite'));
    value_isoffset = any(strcmp(field_validator,'offset'));
    value_isoptional = strcmp(field_validator{1},'optional');

    if (value_isoptional)
        empty = false;

        try
            validateattributes(value,{'numeric'},{'size',[0 0]});
            empty = true;
        catch
        end
        
        if (empty)
            return;
        end
        
        field_validator = field_validator(2:end);
    end
    
    if (value_iscell)
        if (~iscellstr(value) || any(cellfun(@length,value) == 0)) %#ok<ISCLSTR>
            error(['The dataset field ''' field_name ''' is invalid.' newline() 'Expected value to be a cell array of non-empty character vectors.']);
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
        error(['The dataset field ''' field_name ''' is invalid.' newline() strrep(e.message,'Expected input','Expected value')]);
    end
    
    if (value_isfinite)
        value_check = value;
        value_check(isnan(value_check)) = 0;

        try
            validateattributes(value_check,field_type,{'finite'});
        catch
            error(['The dataset field ''' field_name ''' is invalid.' newline() 'Expected input to be finite.']);
        end
    end
    
    if (value_isoffset)
        value_check = value;
        value_check(isnan(value_check)) = 2;

        try
            validateattributes(value_check,field_type,{'>',1});
        catch
            error(['The dataset field ''' field_name ''' is invalid.' newline() 'Expected input to contain non-NaN values > 1.']);
        end
    end

end
