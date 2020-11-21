% [INPUT]
% data = A float t-by-n matrix containing the time series to be converted into rolling windows.
% bw = An integer [2,252] representing the dimension of each rolling window.
% truncate = A boolean that indicates whether to exclude all the rolling windows with a dimension less than the bandwidth (optional, default=false).
%
% [OUTPUT]
% windows = A column cell array of bw-by-n float matrices representing the rolling windows.
%
% [NOTES]
% If the number of observations is less than or equal to the specified bandwidth, a single rolling window containing all the observations is returned.

function windows = extract_rolling_windows(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data',@(x)validateattributes(x,{'double'},{'2d' 'nonempty'}));
        ip.addRequired('bw',@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 2 'scalar'}));
        ip.addOptional('truncate',false,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    data = ipr.data;
    bw = ipr.bw;
    truncate = ipr.truncate;
    
    nargoutchk(1,1);
    
    windows = extract_rolling_windows_internal(data,bw,truncate);

end

function windows = extract_rolling_windows_internal(data,bw,truncate)

    t = size(data,1);
    
    if (bw >= t)
        windows = cell(1,1);
        windows{1} = data;
        return;
    end
    
    limit = t - bw + 1;

    if (truncate)
        windows = cell(limit,1);

        for i = 1:limit
            windows{i} = data(i:bw+i-1,:);
        end
    else
        windows = cell(t,1);
        
        k = max(round(nthroot(bw,1.81),0),5);

        for i = 1:(bw - 1)
            windows{i} = data(1:max(i,k),:);
        end
        
        for i = 1:limit
            windows{i+bw-1} = data(i:bw+i-1,:);
        end
    end

end
