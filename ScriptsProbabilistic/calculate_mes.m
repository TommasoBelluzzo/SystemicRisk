% [INPUT]
% ret0_m = A vector of floats containing the demeaned market index log returns.
% s_m    = A vector of floats containing the volatilities of the market index log returns.
% ret0_x = A vector of floats containing the demeaned firm log returns.
% s_x    = A vector of floats containing the volatilities of the firm log returns.
% beta_x = A vector of floats containing the firm CAPM betas.
% p_mx   = A vector of floats containing the DCC coefficients.
% a      = A float [0.01,0.10] representing the complement to 1 of the confidence level (optional, default=0.05).
% d      = A float representing the six-month crisis threshold for the market index decline used to calculate LRMES (optional, default=0.40).
%
% [OUTPUT]
% mes    = A vector of floats containing the MES values.
% lrmes  = A vector of floats containing the LRMES approximated values.

function [mes,lrmes] = calculate_mes(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('ret0_m',@(x)validateattributes(x,{'double','single'},{'vector','finite','nonempty','nonnan','real'}));
        ip.addRequired('s_m',@(x)validateattributes(x,{'double','single'},{'vector','finite','nonempty','nonnan','real'}));
        ip.addRequired('ret0_x',@(x)validateattributes(x,{'double','single'},{'vector','finite','nonempty','nonnan','real'}));
        ip.addRequired('s_x',@(x)validateattributes(x,{'double','single'},{'vector','finite','nonempty','nonnan','real'}));
        ip.addRequired('beta_x',@(x)validateattributes(x,{'double','single'},{'vector','finite','nonempty','nonnan','real'}));
        ip.addRequired('p_mx',@(x)validateattributes(x,{'double','single'},{'vector','finite','nonempty','nonnan','real'}));
        ip.addOptional('a',0.05,@(x)validateattributes(x,{'double','single'},{'scalar','real','finite','>=',0.01,'<=',0.10}));
        ip.addOptional('d',0.40,@(x)validateattributes(x,{'double','single'},{'scalar','real','finite','>=',0.05,'<=',0.99}));
    end

    ip.parse(varargin{:});
    ip_res = ip.Results;

    [mes,lrmes] = calculate_mes_internal(ip_res.ret0_m,ip_res.s_m,ip_res.ret0_x,ip_res.s_x,ip_res.beta_x,ip_res.p_mx,ip_res.a,ip_res.d);

end

function [mes,lrmes] = calculate_mes_internal(ret0_m,s_m,ret0_x,s_x,beta_x,p_mx,a,d)

    c = quantile(ret0_m,a);
    h = 1 * (length(ret0_m) ^ (-0.2));
    u = ret0_m ./ s_m;

    x_den = sqrt(1 - (p_mx .^ 2));
    x_num = (ret0_x ./ s_x) - (p_mx .* u);
    x = x_num ./ x_den;

    f = normcdf(((c ./ s_m) - u) ./ h);
    k1 = sum(u .* f) ./ sum(f);
    k2 = sum(x .* f) ./ sum(f);

    mes = (s_x .* p_mx .* k1) + (s_x .* x_den .* k2);
    lrmes = 1 - exp(log(1 - d) .* beta_x);

end
