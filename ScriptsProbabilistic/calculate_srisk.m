% [INPUT]
% lrmes = A column vector containing the Long Run MES (LRMES) values.
% dx    = A column vector containing the firm total liabilities.
% ex    = A column vector containing the firm market capitalization.
% l     = A scalar [0,1] representing the capital adequacy ratio.
%
% [OUTPUT]
% srisk   = A column vector containing the SRISK values.

function srisk = calculate_srisk(lrmes,dx,ex,l)

    srisk = (l .* dx) - ((1 - l) .* (1 - lrmes) .* ex);

end