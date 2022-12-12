% [INPUT]
% data = A float t-by-n matrix containing the time series to be distressed.
% offsets = A vector of length n containing the distress begin offsets of each firm; offsets must be equal to NaN in absence of distress.
%
% [OUTPUT]
% ts = A float t-by-n matrix containing the original time series in which distressed observations are replaced with NaNs.

function data = distress_data(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data',@(x)validateattributes(x,{'double'},{'real' '2d'}));
        ip.addRequired('offsets',@(x)validateattributes(x,{'double'},{'real' 'vector' 'nonempty'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    [data,offsets] = validate_input(ipr.data,ipr.offsets);

    nargoutchk(1,1);

    data = distress_data_internal(data,offsets);

end

function data = distress_data_internal(data,offsets)

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

function [data,offsets] = validate_input(data,offsets)

    if (~isempty(data))
        n = size(data,2);
        k = numel(offsets);

        if (n ~= k)
            error(['The number of columns in the time series (' num2str(n) ') must be equal to the number of offsets (' num2str(k) ').']);
        end
    end

end
