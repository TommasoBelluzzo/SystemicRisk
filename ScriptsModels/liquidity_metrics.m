% [INPUT]
% p = A vector of floats [0,Inf) of length t representing the prices.
% r = A vector of floats (-Inf,Inf) of length t representing the logarithmic returns.
% v = A vector of floats [0,Inf) of length t representing the trading volumes.
% cp = A vector of floats [0,Inf) of length t representing the market capitalization.
% bwl = An integer [90,252] representing the dimension of the long bandwidth.
% bwm = An integer [21,90) representing the dimension of the medium bandwidth.
% bws = An integer [5,21) representing the dimension of the short bandwidth.
%
% [OUTPUT]
% hhlr = A column vector of floats [0,Inf) of length t representing the Hui-Heubel Liquidity Ratio.
% tr = A column vector of floats [0,Inf) of length t representing the Turnover Ratio.
% vr = A column vector of floats [0,Inf) of length t representing the Variance Ratio.

function [hhlr,tr,vr] = liquidity_metrics(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('p',@(x)validateattributes(x,{'double'},{'real' 'finite' 'nonnegative' 'vector' 'nonempty'}));
        ip.addRequired('r',@(x)validateattributes(x,{'double'},{'real' 'finite' 'vector' 'nonempty'}));
        ip.addRequired('v',@(x)validateattributes(x,{'double'},{'real' 'finite' 'nonnegative' 'vector' 'nonempty'}));
        ip.addRequired('cp',@(x)validateattributes(x,{'double'},{'real' 'finite' 'nonnegative' 'vector' 'nonempty'}));
        ip.addRequired('bwl',@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 90 '<=' 252 'scalar'}));
        ip.addRequired('bwm',@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 21 '<' 90 'scalar'}));
        ip.addRequired('bws',@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 5 '<' 21 'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    [p,r,v,cp,bwl,bwm,bws] = validate_input(ipr.p,ipr.r,ipr.v,ipr.cp,ipr.bwl,ipr.bwm,ipr.bws);

    nargoutchk(3,3);

    [hhlr,tr,vr] = liquidity_metrics_internal(p,r,v,cp,bwl,bwm,bws);

end

function [hhlr,tr,vr] = liquidity_metrics_internal(p,r,v,cp,bwl,bwm,bws)

    hhlr = calculate_hhlr(p,v,cp,bwl,bws);
    tr = calculate_tr(v,cp,bwl);
    vr = calculate_vr(r,bwl,bwm);

end

function hhlr = calculate_hhlr(p,v,cp,bwl,bws)

    tr = v ./ cp;
    tr(~isfinite(tr)) = 0;

    windows_p = extract_rolling_windows(p,bws);
    dp = cellfun(@(x)(max(x) - min(x)) / min(x),windows_p);

    alpha = 2 / (bwl + 1);

    hhlr = dp ./ tr;
    hhlr(~isfinite(hhlr)) = 0;
    hhlr(1:bws) = mean(hhlr(bws+1:bws*2+1));
    hhlr = [hhlr(1); filter(alpha,[1 (alpha - 1)],hhlr(2:end),(1 - alpha) * hhlr(1))];
    hhlr = (hhlr - min(hhlr)) ./ (max(hhlr) - min(hhlr));

end

function tr = calculate_tr(v,cp,bwl)

    alpha = 2 / (bwl + 1);

    tr = v ./ cp;
    tr(~isfinite(tr)) = 0;
    tr = [tr(1); filter(alpha,[1 (alpha - 1)],tr(2:end),(1 - alpha) * tr(1))];
    tr = (tr - min(tr)) ./ (max(tr) - min(tr));

end

function vr = calculate_vr(r,bwl,bwm)

    alpha = 2 / (bwl + 1);
    t = bwl / bwm;

    windows_long = extract_rolling_windows(r,bwl);
    var_long = cellfun(@var,windows_long);

    windows_short = extract_rolling_windows(r,bwm);
    var_short = cellfun(@var,windows_short);

    vr = var_long ./ (t .* var_short);
    vr(~isfinite(vr)) = 0;
    vr(1:bwm) = mean(vr(bwm+1:bwm*2+1));
    vr = [vr(1); filter(alpha,[1 (alpha - 1)],vr(2:end),(1 - alpha) * vr(1))];

end

function [p,r,v,cp,bwl,bwm,bws] = validate_input(p,r,v,cp,bwl,bwm,bws)

    data = {p(:) r(:) v(:) cp(:)};

    l = unique(cellfun(@numel,data));

    if (numel(l) ~= 1)
        error('The number of elements of ''p'', ''r'' and ''v'' must be equal.');
    end

    if (l < 5)
        error('The value of ''p'', ''r'' and ''v'' is invalid. Expected inputs to be vectors containing at least 5 elements.');
    end

    [p,r,v,cp] = deal(data{:});

    if (bwl < (bwm * 2))
        error(['The long bandwidth (' num2str(bwl) ') must be at least twice the medium bandwidth (' num2str(bwm) ').']);
    end

    if (bwm < (bws * 2))
        error(['The medium bandwidth (' num2str(bwm) ') must be at least twice the short bandwidth (' num2str(bws) ').']);
    end

end
