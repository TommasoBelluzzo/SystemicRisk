% [INPUT]
% rdm  = A column vector containing the demeaned market index log returns.
% sm   = A column vector containing the volatilities of the market index log returns.
% rdx  = A column vector containing the demeaned firm log returns.
% sx   = A column vector containing the volatilities of the firm log returns.
% pmx  = A column vector containing the DCC coefficients.
% k    = A scalar [0,1] representing the confidence level.
%
% [OUTPUT]
% mes   = A column vector containing the MES values.
% lrmes = A column vector containing the Long Run MES (LRMES) approximated values.

function [mes,lrmes] = calculate_mes(rdm,sm,rdx,sx,pmx,k)

    c = quantile(rdm,k);

    um = rdm ./ sm;
    h = 1 * (length(rdm) ^ (-0.2));
    
    xden = sqrt(1 - (pmx .^ 2));
    x = ((rdx ./ sx) - (pmx .* um)) ./ xden;

    f = normcdf(((c ./ sm) - um) ./ h);
    k1 = sum(um .* f) ./ sum(f);
    k2 = sum(x .* f) ./ sum(f);

    mes = ((sx .* pmx .* k1) + (sx .* xden .* k2)) .* -1;
    lrmes = 1 - exp(-18 .* mes);

end