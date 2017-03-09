% [INPUT]
% rdem_m = A column vector containing the demeaned market index log returns.
% rdem_x = A column vector containing the demeaned firm log returns.
% var_x  = A column vector containing the firm unconditional VaR.
% k      = A scalar [0,1] representing the confidence level.
% svars  = A t-by-n matrix containing the lagged state variables (optional, default=[]).
%          Example of state variables to use for S&P500 and U.S. market:
%           - Dow Jones U.S. Select Real Estate Securities Index (RESI)
%           - Volatility Index (VIX)
%           - Variations of 3M TBR
%           - Credit Spread (BAA CBR minus 10Y TBR)
%           - Liquidity Spread (3M FFR minus 3M TBR)
%           - Yield Spread (10Y TBR minus 3M TBR)
%
% [OUTPUT]
% covar  = A column vector containing the CoVaR values.
% dcovar = A column vector containing the Delta CoVaR values.

function [covar,dcovar] = calculate_covar(rdem_m,rdem_x,var_x,k,svars)

    t = length(rdem_m);

    if (nargin < 5)
        [beta,~,~,~] = quantile_regression(rdem_m,rdem_x,k);
        covar = (beta(1) + (beta(2) .* var_x)) .* -1;
    else
        [beta,~,~,~] = quantile_regression(rdem_m,[rdem_x svars],k);
        covar = beta(1) + (beta(2) .* var_x);

        svars_cnt = size(svars,2);
        
        for i = 1:svars_cnt
            covar = covar + (beta(i+2) .* svars(:,i));
        end

        covar = covar .* -1;
    end

    dcovar = (beta(2) .* (var_x - repmat(median(rdem_x),t,1))) .* -1;

end