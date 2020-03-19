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

    validate_field(data,'TimeSeries',{'cellstr'},{'nonempty','size',[1 9]});
    
    validate_field(data,'SupportsComponent',{'logical'},{'scalar'});
    validate_field(data,'SupportsConnectedness',{'logical'},{'scalar'});
    validate_field(data,'SupportsCrossSectional',{'logical'},{'scalar'});
    validate_field(data,'SupportsDefault',{'logical'},{'scalar'});
    validate_field(data,'SupportsSpillover',{'logical'},{'scalar'});

    n = validate_field(data,'N',{'numeric'},{'scalar','integer','real','finite','>=',3});
    t = validate_field(data,'T',{'numeric'},{'scalar','integer','real','finite','>=',252});

    validate_field(data,'DatesNum',{'numeric'},{'integer','real','finite','>',0,'nonempty','size',[t 1]});
    validate_field(data,'DatesStr',{'cellstr'},{'nonempty','size',[t 1]});
    validate_field(data,'MonthlyTicks',{'logical'},{'scalar'});

    validate_field(data,'IndexName',{'char'},{'nonempty','size',[1 NaN]});
    validate_field(data,'IndexReturns',{'double'},{'real','finite','nonempty','size',[t 1]});

    validate_field(data,'FirmNames',{'cellstr'},{'nonempty','size',[1 n]});
    validate_field(data,'FirmReturns',{'double'},{'real','finite','nonempty','size',[t n]});
    validate_field(data,'FirmDefaults',{'double'},{'real','nonempty','size',[1 n]});

    validate_field(data,'Capitalization',{'double'},{'optional','real','finite','nonnegative','nonempty','size',[t n]});
    validate_field(data,'CapitalizationLagged',{'double'},{'optional','real','finite','nonnegative','nonempty','size',[t n]});

    validate_field(data,'CDS',{'double'},{'optional','real','finite','nonnegative','nonempty','size',[t n]});
    validate_field(data,'RiskFreeRate',{'double'},{'optional','real','finite','nonempty','size',[t 1]});

    validate_field(data,'Assets',{'double'},{'optional','real','finite','nonnegative','nonempty','size',[t n]});
    validate_field(data,'Equity',{'double'},{'optional','real','finite','nonempty','size',[t n]});
    validate_field(data,'Liabilities',{'double'},{'optional','real','finite','nonnegative','nonempty','size',[t n]});
    validate_field(data,'LiabilitiesRolled',{'double'},{'optional','real','finite','nonnegative','nonempty','size',[t n]});
    validate_field(data,'SeparateAccounts',{'double'},{'optional','real','finite','nonnegative','nonempty','size',[t n]});

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

    if (strcmp(field_validator{1},'optional'))
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
    end

    try
        validateattributes(value,field_type,field_validator);
    catch e
        error(['The dataset field ''' field_name ''' is invalid.' newline() strrep(e.message,'Expected input','Expected value')]);
    end

end
