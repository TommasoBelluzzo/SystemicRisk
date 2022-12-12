% [INPUT]
% r = A float t-by-n matrix (-Inf,Inf) representing the logarithmic returns.
% f = A float [0.2,0.8] representing the percentage of components to include in the computation of the Absorption Ratio (optional, default=0.2).
%
% [OUTPUT]
% ar = A float [0,1] representing the Absorption Ratio.
% cs = A float [0,Inf) representing the Correlation Surprise.
% ti = A float [0,Inf) representing the Turbulence Index.

function [ar,cs,ti] = component_metrics(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('r',@(x)validateattributes(x,{'double'},{'real' '2d' 'nonempty'}));
        ip.addOptional('f',0.2,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.2 '<=' 0.8 'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    [r,fc] = validate_input(ipr.r,ipr.f);

    nargoutchk(3,3);

    [ar,cs,ti] = component_metrics_internal(r,fc);

end

function [ar,cs,ti] = component_metrics_internal(r,fc)

    zero_indices = find(~r);
    r(zero_indices) = (-9e-9 .* rand(numel(zero_indices),1)) + 1e-8;

    novar_indices = find(var(r,1) == 0);
    r(:,novar_indices) = r(:,novar_indices) + ((-9e-9 .* rand(size(r(:,novar_indices)))) + 1e-8);

    c = cov(r);
    c_size = size(c);

    bm = zeros(c_size);
    bm(logical(eye(size(c)))) = diag(c);

    e = eigs(c,size(c,1));

    v = r(end,:) - mean(r(1:end-1,:),1);
    vt = v.';

    ar = sum(e(1:fc)) / trace(c);
    cs = ((v / c) * vt) / ((v / bm) * vt);
    ti = (v / c) * vt;

end

function [r,fc] = validate_input(r,f)

    nan_indices = any(isnan(r),1);
    r(:,nan_indices) = [];

    [t,n] = size(r);

    if ((t < 5) || (n < 2))
        error('The value of ''r'' is invalid. Expected input to be a matrix with a minimum size of 5x2, after the exclusion of time series containing NaN values.');
    end

    fc = max(round(n * f,0),1);

end
