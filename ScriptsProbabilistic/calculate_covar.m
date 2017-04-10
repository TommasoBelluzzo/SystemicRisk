% [INPUT]
% ret0_m = A vector of floats containing the demeaned market index log returns.
% ret0_x = A vector of floats containing the demeaned firm log returns.
% var_x  = A vector of floats containing the firm unconditional VaR.
% a      = A float [0.01,0.10] representing the complement to 1 of the confidence level (optional, default=0.05).
% svars  = A numeric t-by-n matrix containing the lagged state variables (optional, default=[]).
%          Example of state variables to use for S&P500 and U.S. market:
%           - Dow Jones U.S. Select Real Estate Securities Index (RESI)
%           - Volatility Index (VIX)
%           - Variations of 3M TBR
%           - Credit Spread (BAA CBR minus 10Y TBR)
%           - Liquidity Spread (3M FFR minus 3M TBR)
%           - Yield Spread (10Y TBR minus 3M TBR)
%
% [OUTPUT]
% covar  = A vector of floats containing the firm CoVaR.
% dcovar = A vector of floats containing the firm Delta CoVaR.

function [covar,dcovar] = calculate_covar(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('ret0_m',@(x)validateattributes(x,{'double','single'},{'vector','finite','nonempty','nonnan','real'}));
        ip.addRequired('ret0_x',@(x)validateattributes(x,{'double','single'},{'vector','finite','nonempty','nonnan','real'}));
        ip.addRequired('var_x',@(x)validateattributes(x,{'double','single'},{'vector','finite','nonempty','nonnan','real'}));
        ip.addOptional('a',0.05,@(x)validateattributes(x,{'double','single'},{'scalar','real','finite','>=',0.01,'<=',0.10}));
        ip.addOptional('svars',[],@(x)validateattributes(x,{'numeric'},{'2d','finite','nonnan','real'}));
    end

    ip.parse(varargin{:});
    ip_res = ip.Results;

    [covar,dcovar] = calculate_covar_internal(ip_res.ret0_m,ip_res.ret0_x,ip_res.var_x,ip_res.a,ip_res.svars);

end

function [covar,dcovar] = calculate_covar_internal(ret0_m,ret0_x,var_x,a,svars)

    if (isempty(svars))
        beta = quantile_regression(ret0_m,ret0_x,a);
        covar = beta(1) + (beta(2) .* var_x);
    else
        beta = quantile_regression(ret0_m,[ret0_x svars],a);
        covar = beta(1) + (beta(2) .* var_x);

        svars_cnt = size(svars,2);
        
        for i = 1:svars_cnt
            covar = covar + (beta(i+2) .* svars(:,i));
        end
    end

    dcovar = beta(2) .* (var_x - repmat(median(ret0_x),length(ret0_m),1));

end

function beta = quantile_regression(y,x,k)

    [xn,xm] = size(x);

    x = [ones(xn,1) x];
    xm = xm + 1;
    xs = x;

    beta = ones(xm,1);

    diff = 1;
    iter = 0;

    while ((diff > 1e-6) && (iter < 1000))
        xst = xs';
        beta_0 = beta;

        beta = ((xst * x) \ xst) * y;

        rsd = y - (x * beta);
        rsd(abs(rsd)<0.000001) = 0.000001;
        rsd(rsd<0) = k * rsd(rsd<0);
        rsd(rsd>0) = (1 - k) * rsd(rsd>0);
        rsd = abs(rsd);

        z = zeros(xn,xm);

        for i = 1:xm 
            z(:,i) = x(:,i) ./ rsd;
        end

        xs = z;
        beta_1 = beta;
        
        diff = max(abs(beta_1 - beta_0));
        iter = iter + 1;
    end

end