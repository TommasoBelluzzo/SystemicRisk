% [INPUT]
% rdm  = A column vector containing the demeaned market index log returns.
% rdx  = A column vector containing the demeaned firm log returns.
% varx = A column vector containing firm unconditional VaR.
% k    = A scalar [0,1] representing the confidence level.
% sv   = A t-by-n matrix containing the lagged state variables (optional, default=[]).
%        Example of state variables to use for S&P500 and U.S. market:
%         - Dow Jones U.S. Select Real Estate Securities Index (RESI)
%         - Volatility Index (VIX)
%         - Variations of 3M TBR
%         - Credit Spread (BAA CBR minus 10Y TBR)
%         - Liquidity Spread (3M FFR minus 3M TBR)
%         - Yield Spread (10Y TBR minus 3M TBR)
%
% [OUTPUT]
% covar  = A column vector containing the CoVaR values.
% dcovar = A column vector containing the Delta CoVaR values.

function [covar,dcovar] = calculate_covar(rdm,rdx,varx,k,sv)

    if (nargin < 4)
        error('The function requires at least 4 arguments.');
    end

    t = length(rdm);

    if (nargin < 5)
        [beta,~,~,~] = quantile_regression(rdm,rdx,k);
        covar = (beta(1) + (beta(2) .* varx)) .* -1;
    else
        [beta,~,~,~] = quantile_regression(rdm,[rdx sv],k);
        covar = beta(1) + (beta(2) .* varx);

        for i = 3:length(beta)
            covar = covar + (beta(i) .* sv(:,(i - 2)));
        end

        covar = covar .* -1;
    end

    dcovar = (beta(2) .* (varx - repmat(median(rdx),t,1))) .* -1;

end