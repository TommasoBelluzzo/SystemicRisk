% [INPUT]
% ds = A float t-by-n matrix containing the time series.
% offsets = A vector of length n containing the numeric date of the distress begin of each firm; offsets are equal to NaN in absence of distress.
%
% [OUTPUT]
% ds = A float t-by-n matrix containing the original time series in which distressed observations are replaced with NaNs.

function ds = distress_data(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('ds',@(x)validateattributes(x,{'double'},{'real','2d'}));
        ip.addRequired('offsets',@(x)validateattributes(x,{'double'},{'real','vector','nonempty'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    [ds,offsets] = validate_input(ipr.ds,ipr.offsets);

    nargoutchk(1,1);

    ds = distress_data_internal(ds,offsets);

end

function ds = distress_data_internal(ds,offsets)

    if (isempty(ds))
        return;
    end

    for i = 1:numel(offsets)
        offset = offsets(i);

        if (isnan(offset))
            continue;
        end

        ds(offset:end,i) = NaN;
    end

end

function [ds,offsets] = validate_input(ds,offsets)

    n = size(ds,2);
    
    if ((n ~= 0) && (n ~= numel(offsets)))
        error('The number of columns in the time series must be equal to the number of firms.');
    end
    
end
