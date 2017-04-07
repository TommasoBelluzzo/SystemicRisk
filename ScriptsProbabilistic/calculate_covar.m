% [INPUT]
% ret0_m = A vector of floats containing the demeaned market index log returns.
% ret0_x = A vector of floats containing the demeaned firm log returns.
% s_x    = A vector of floats containing the volatilities of the firm log returns.
% a      = A float representing the complement to 1 of the confidence level (optional, default=0.05).
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
% var    = A vector of floats containing the firm unconditional VaR.
% covar  = A vector of floats containing the firm CoVaR.
% dcovar = A vector of floats containing the firm Delta CoVaR.

function [var,covar,dcovar] = calculate_covar(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('ret0_m',@(x)validateattributes(x,{'double','single'},{'vector','finite','nonempty','nonnan','real'}));
        ip.addRequired('ret0_x',@(x)validateattributes(x,{'double','single'},{'vector','finite','nonempty','nonnan','real'}));
        ip.addRequired('s_x',@(x)validateattributes(x,{'double','single'},{'vector','finite','nonempty','nonnan','real'}));
        ip.addOptional('a',0.05,@(x)validateattributes(x,{'double','single'},{'scalar','real','finite','>=',0.01,'<=',0.10}));
        ip.addOptional('svars',[],@(x)validateattributes(x,{'numeric'},{'2d','finite','nonnan','real'}));
    end

    ip.parse(varargin{:});
    ip_res = ip.Results;

    [var,covar,dcovar] = calculate_covar_internal(ip_res.ret0_m,ip_res.ret0_x,ip_res.s_x,ip_res.a,ip_res.svars);

end

function [var,covar,dcovar] = calculate_covar_internal(ret0_m,ret0_x,s_x,a,svars)

    t = length(ret0_m);

    var = s_x * quantile((ret0_x ./ s_x),a);

    if (isempty(svars))
        [beta,~,~,~] = quantile_regression(ret0_m,ret0_x,a);
        covar = (beta(1) + (beta(2) .* var)) .* -1;
    else
        [beta,~,~,~] = quantile_regression(ret0_m,[ret0_x svars],a);
        covar = beta(1) + (beta(2) .* var);

        svars_cnt = size(svars,2);
        
        for i = 1:svars_cnt
            covar = covar + (beta(i+2) .* svars(:,i));
        end

        covar = covar .* -1;
    end

    if (nargout > 2)
        dcovar = (beta(2) .* (var - repmat(median(ret0_x),t,1))) .* -1;
        var = var .* -1;
    else
        var = var .* -1;
    end

end
