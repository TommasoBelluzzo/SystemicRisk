% [INPUT]
% lrmes = A column vector containing the Long Run MES (LRMES) values.
% tl_x  = A column vector containing the firm total liabilities.
% mc_x  = A column vector containing the firm market capitalization.
% l     = A scalar [0,1] representing the capital adequacy ratio.
%
% [OUTPUT]
% srisk = A column vector containing the SRISK values.

function srisk = calculate_srisk(lrmes,tl_x,mc_x,l)

    srisk = (l .* tl_x) - ((1 - l) .* (1 - lrmes) .* mc_x);
    srisk(srisk<0) = 0;

end