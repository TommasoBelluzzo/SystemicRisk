% [INPUT]
% offsets = A vector of length n containing the distress beginning offset of each firm; offsets are equal to NaN in absence of distress.
% data = A float t-by-n matrix containing the time series.
%
% [OUTPUT]
% data = A float t-by-n matrix containing the original time series in which distressed observations are replaced with NaNs.

function data = distress_data(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('offsets',@(x)validateattributes(x,{'double'},{'real','vector','nonempty'}));
        ip.addRequired('data',@(x)validateattributes(x,{'double'},{'real','2d'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    offsets = ipr.offsets;
    data = ipr.data;
    
    n = size(data,2);
    
    if ((n ~= 0) && (n ~= numel(offsets)))
        error('The number of columns in the time series must be equal to the number of firms.');
    end
    
    nargoutchk(1,1);

    data = distress_data_internal(offsets,data);

end

function data = distress_data_internal(offsets,data)

    if (isempty(data))
        return;
    end

    for i = 1:numel(offsets)
        offset = offsets(i);

        if (isnan(offset))
            continue;
        end

        data(offset:end,i) = NaN;
    end

end