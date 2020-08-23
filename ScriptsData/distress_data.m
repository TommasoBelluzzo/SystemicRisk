% [INPUT]
% ts = A float t-by-n matrix containing the time series.
% offsets = A vector of length n containing the numeric date of the distress begin of each firm; offsets are equal to NaN in absence of distress.
%
% [OUTPUT]
% ts = A float t-by-n matrix containing the original time series in which distressed observations are replaced with NaNs.

function ts = distress_data(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('ts',@(x)validateattributes(x,{'double'},{'real' '2d'}));
        ip.addRequired('offsets',@(x)validateattributes(x,{'double'},{'real' 'vector' 'nonempty'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    [ts,offsets] = validate_input(ipr.ts,ipr.offsets);

    nargoutchk(1,1);

    ts = distress_data_internal(ts,offsets);

end

function ts = distress_data_internal(ts,offsets)

    if (isempty(ts))
        return;
    end

    for i = 1:numel(offsets)
        offset = offsets(i);

        if (isnan(offset))
            continue;
        end

        ts(offset:end,i) = NaN;
    end

end

function [ts,offsets] = validate_input(ts,offsets)

    n = size(ts,2);
    k = numel(offsets);
    
    if ((n ~= 0) && (n ~= numel(offsets)))
        error(['The number of columns in the time series (' num2str(n) ') must be equal to the number of offsets (' num2str(k) ').']);
    end
    
end
