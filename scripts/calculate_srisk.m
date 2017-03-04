% [INPUT]
% lrmes = A column vector containing the Long Run MES (LRMES) values.
% dx    = A column vector containing the firm total liabilities.
% ex    = A column vector containing the firm market capitalization.
% l     = A scalar [0,1] representing the capital adequacy ratio (optional, default=0.08).
% [OUTPUT]
% srisk   = A column vector containing the SRISK values.

function srisk = calculate_srisk(lrmes,dx,ex,l)

    if (nargin < 3)
        error('The function requires at least 3 arguments.');
    end

    if (nargin < 4)
        l = 0.08;
    end

    srisk = (l .* dx) - ((1 - l) .* (1 - lrmes) .* ex);

end