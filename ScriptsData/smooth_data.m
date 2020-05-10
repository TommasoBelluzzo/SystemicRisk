% [INPUT]
% data = A float t-by-n matrix containing the time series to be smooted.
% s = An integer [1,t] representing the span of the smoothing filter (optional, default=21).
%
% [OUTPUT]
% data = A float t-by-n matrix containing the smoothed time series.

function data = smooth_data(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data',@(x)validateattributes(x,{'double'},{'real','2d','nonempty'}));
        ip.addOptional('s',21,@(x)validateattributes(x,{'double'},{'real','finite','integer','>=',1,'scalar'}));
    end

    ip.parse(varargin{:});
    
    ipr = ip.Results;
    [data,s] = validate_input(ipr.data,ipr.s);

    nargoutchk(1,1);

    data = smooth_data_internal(data,s);

end

function data = smooth_data_internal(data,s)

    w = s + mod(s,2) - 1;
    
    for i = 1:size(data,2)
        data(:,i) = smoothing_function(data(:,i),s,w);
    end

end

function ys = smoothing_function(y,s,w)

    nan_indices = isnan(y);
    nan_found = any(nan_indices);

	if (~nan_found)
        if (w == 1)
            ys = y;
            return;
        end

        n = numel(y);
        z = filter(ones(w,1) ./ w,1,y);
        
        ys_begin = cumsum(y(1:w-2));
        ys_begin = ys_begin(1:2:end) ./ (1:2:w-2).';
        
        ys_end = cumsum(y(n:-1:n-w+3));
        ys_end = ys_end(end:-2:1)./(w-2:-2:1)';

        ys = [ys_begin; z(w:end); ys_end];
    else
        z1 = y;
        z1(nan_indices) = 0;

        z2 = double(~nan_indices);

        ys = smoothing_function(z1,s,w) ./ smoothing_function(z2,s,w);
	end

end

function [data,s] = validate_input(data,s)

    if (isscalar(data))
        error('The value of ''data'' is invalid. Expected input to be a vector or a 2d matrix.');
    end

    [t,n] = size(data);
    
    if ((any([t,n]) == 1) && (n > t))
        error('The value of ''data'' is invalid. Expected input to be a column vector when a single time series is defined.');
    end
    
    if (s > t)
        error(['The value of ''s'' is invalid. Expected input to be less than or equal to ' num2str(t) '.']);
    end

end
