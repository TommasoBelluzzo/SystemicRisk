% [INPUT]
% vd = A float n-by-n matrix (-Inf,Inf) representing the variance decomposition.
%
% [OUTPUT]
% sf = A row vector of floats [0,Inf) of length n representing the Spillovers From.
% sf = A row vector of floats [0,Inf) of length n representing the Spillovers To.
% sf = A row vector of floats (-Inf,Inf) of length n representing the Net Spillovers.
% si = A float (-Inf,Inf) representing the Spillover Index.

function [sf,st,sn,si] = spillover_metrics(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('vd',@(x)validateattributes(x,{'double'},{'real' 'finite' '2d' 'square' 'nonempty'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    vd = validate_input(ipr.vd);

    nargoutchk(4,4);

    [sf,st,sn,si] = spillover_metrics_internal(vd);

end

function [sf,st,sn,si] = spillover_metrics_internal(vd)

    vd_diag = diag(vd);

    sf = min(max(sum(vd,2) - vd_diag,0),1);
    st = sum(vd,1).' - vd_diag;
    sn = st - sf;

    si = sum(sf,1) / (sum(vd_diag) + sum(sf,1));

    sf = sf.';
    st = st.';
    sn = sn.';

end

function vd = validate_input(vd)

    vdv = vd(:);

    if (numel(vdv) < 4)
        error('The value of ''vd'' is invalid. Expected input to be a square matrix with a minimum size of 2x2.');
    end

    tol = 1e4 * eps(max(abs(vdv))); 

    if (any(abs(sum(vd,2) - 1) > tol))
        error('The value of ''vd'' is invalid. Expected input to be a matrix whose row sums are equal to 1.');
    end

end
