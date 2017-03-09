% [INPUT]
% rdem_m = A column vector containing the demeaned market index log returns.
% s_m    = A column vector containing the volatilities of the market index log returns.
% rdem_x = A column vector containing the demeaned firm log returns.
% s_x    = A column vector containing the volatilities of the firm log returns.
% rho    = A column vector containing the DCC coefficients.
% k      = A scalar [0,1] representing the confidence level.
%
% [OUTPUT]
% mes    = A column vector containing the MES values.
% lrmes  = A column vector containing the Long Run MES (LRMES) approximated values.

function [mes,lrmes] = calculate_mes(rdem_m,s_m,rdem_x,s_x,rho,k)

    c = quantile(rdem_m,k);
    h = 1 * (length(rdem_m) ^ (-0.2));
    u = rdem_m ./ s_m;

    x_den = sqrt(1 - (rho .^ 2));
    x_num = (rdem_x ./ s_x) - (rho .* u);
    x = x_num ./ x_den;

    f = normcdf(((c ./ s_m) - u) ./ h);
    k1 = sum(u .* f) ./ sum(f);
    k2 = sum(x .* f) ./ sum(f);

    mes = ((s_x .* rho .* k1) + (s_x .* x_den .* k2)) .* -1;
    lrmes = 1 - exp(-18 .* mes);

end