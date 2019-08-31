% [INPUT]
% data = A numeric t-by-n matrix containing the time series.
% bandwidth = An integer greater than or equal to 2 representing the bandwidth (dimension) of each rolling window.
%
% [OUTPUT]
% windows = A vector of numeric bandwidth-by-n matrices representing the rolling windows.
%
% [NOTES]
% If the number of observations is less than or equal to the specified bandwidth, a single rolling window containing all the observations is returned.

function windows = extract_rolling_windows(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data',@(x)validateattributes(x,{'numeric'},{'2d','nonempty'}));
        ip.addRequired('bandwidth',@(x)validateattributes(x,{'numeric'},{'scalar','integer','real','finite','>=',2}));
    end

    ip.parse(varargin{:});
    ipr = ip.Results;
    
    windows = get_rolling_windows_internal(ipr.data,ipr.bandwidth);

end

function windows = get_rolling_windows_internal(data,bandwidth)

    t = size(data,1);
    
    if (bandwidth >= t)
        windows = cell(1,1);
        windows{1} = data;
        return;
    end

    limit = t - bandwidth + 1;
    windows = cell(limit,1);

    for i = 1:limit
        windows{i} = data(i:bandwidth+i-1,:);
    end

end