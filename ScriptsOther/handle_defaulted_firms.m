% [INPUT]
% data = A numeric t-by-n matrix containing the time series.
% firm_defaults = A vector containing the default offset of each firm; offsets of alive firms are set to NaN.
%
% [OUTPUT]
% data = A numeric t-by-n matrix containing the original time series with NaNs replacing the values of defaulted firms.

function data = handle_defaulted_firms(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data',@(x)validateattributes(x,{'numeric'},{'2d','nonempty','real'}));
        ip.addRequired('firm_defaults',@(x)validateattributes(x,{'numeric'},{'vector','nonempty','real'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    data = ipr.data;
    firm_defaults = ipr.firm_defaults;
    
    if (size(data,2) ~= numel(firm_defaults))
        error('The number of columns in the time series must be equal to the number of firms.');
    end
    
    nargoutchk(1,1);

    data = handle_defaulted_firms_internal(data,firm_defaults);

end

function data = handle_defaulted_firms_internal(data,firm_defaults)

    for i = 1:numel(firm_defaults)
        firm_default = firm_defaults(i);
        
        if (isnan(firm_default))
            continue;
        end
        
        data(firm_default:end,i) = NaN;
    end

end