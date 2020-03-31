% [INPUT]
% offsets = A vector containing the distress offset of each firm; offsets are set to NaN in absence of distress.
% data = A numeric t-by-n matrix containing the time series.
% lagged = A boolean indicating whether the time series is lagged.
%
% [OUTPUT]
% data = A numeric t-by-n matrix containing the original time series in which distressed observations are replaced with NaNs.

function data = handle_firms_distress(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('offsets',@(x)validateattributes(x,{'numeric'},{'vector','nonempty','real'}));
        ip.addRequired('data',@(x)validateattributes(x,{'numeric'},{'2d','real'}));
        ip.addRequired('lagged',@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    offsets = ipr.offsets;
    data = ipr.data;
    lagged = ipr.lagged;
    
    n = size(data,2);
    
    if ((n ~= 0) && (n ~= numel(offsets)))
        error('The number of columns in the time series must be equal to the number of firms.');
    end
    
    nargoutchk(1,1);

    data = handle_firms_distress_internal(offsets,data,lagged);

end

function data = handle_firms_distress_internal(offsets,data,lagged)

    if (isempty(data))
        return;
    end

    for i = 1:numel(offsets)
        offset = offsets(i);

        if (isnan(offset))
            continue;
        end
        
        if (lagged)
            offset = offset - 1;
        end

        data(offset:end,i) = NaN;
    end

end