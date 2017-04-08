% [INPUT]
% data = A numeric t-by-n matrix containing the time series.
% bw   = An integer representing the bandwidth (dimension) of each rolling window.
%
% [OUTPUT]
% win  = A vector of numeric bw-by-n matrices representing the rolling windows.
%
% [NOTES]
% If the number of observations is less than or equal to the specified bandwidth, a single rolling window containing all the observations is returned.

function win = get_rolling_windows(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data',@(x)validateattributes(x,{'numeric'},{'2d','nonempty'}));
        ip.addRequired('bw',@(x)validateattributes(x,{'numeric'},{'scalar','integer','real','finite','>=',2}));
    end

    ip.parse(varargin{:});
    ip_res = ip.Results;
    
    win = get_rolling_windows_internal(ip_res.data,ip_res.bw);

end

function win = get_rolling_windows_internal(data,bw)

    t = size(data,1);
    
    if (bw >= t)
        win = cell(1,1);
        win{1} = data;
        return;
    end

    lim = t - bw + 1;
    win = cell(lim,1);

    for i = 1:lim
        win{i} = data(i:bw+i-1,:);
    end

end