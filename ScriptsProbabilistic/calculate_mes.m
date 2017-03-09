% [INPUT]
% rdem_m = A column vector containing the demeaned market index log returns.
% sm     = A column vector containing the volatilities of the market index log returns.
% rdem_x = A column vector containing the demeaned firm log returns.
% sx     = A column vector containing the volatilities of the firm log returns.
% rho    = A column vector containing the DCC coefficients.
% k      = A scalar [0,1] representing the confidence level.
%
% [OUTPUT]
% mes    = A column vector containing the MES values.
% lrmes  = A column vector containing the Long Run MES (LRMES) approximated values.

function [mes,lrmes] = calculate_mes(rdem_m,sm,rdem_x,sx,rho,k)

    c = quantile(rdem_m,k);
    h = 1 * (length(rdem_m) ^ (-0.2));
    u = rdem_m ./ sm;

    x_den = sqrt(1 - (rho .^ 2));
    x_num = (rdem_x ./ sx) - (rho .* u);
    x = x_num ./ x_den;

    f = normcdf(((c ./ sm) - u) ./ h);
    k1 = sum(u .* f) ./ sum(f);
    k2 = sum(x .* f) ./ sum(f);

    mes = ((sx .* rho .* k1) + (sx .* x_den .* k2)) .* -1;
    lrmes = 1 - exp(-18 .* mes);

end