% [INPUT]
% ret = A t-by-n matrix containing firms log returns.
% bw  = The bandwidth (dimension) of each rolling window.
%
% [OUTPUT]
% win = A column vector containing the rolling windows.
%
% [NOTE]
% If the number of observations is less than or equal to the specified bandwidth, a single rolling window containing all the observations is returned.

function win = get_rolling_windows(ret,bw)

    t = length(ret);
    
    if (bw >= t)
        win = cell(1,1);
        win{1} = ret;
        return;
    end

    lim = t - bw + 1;
    win = cell(lim,1);

    for i = 1:lim
        win{i} = ret(i:bw+i-1,:);
    end

end