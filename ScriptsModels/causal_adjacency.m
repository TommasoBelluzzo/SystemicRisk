% [INPUT]
% r = A float t-by-n matrix (-Inf,Inf) representing the logarithmic returns.
% sst = A float (0.0,0.1] representing the statistical significance threshold for the linear Granger-causality test (optional, default=0.05).
% rp = A boolean indicating whether to use robust p-values for the linear Granger-causality test (optional, default=false).
%
% [OUTPUT]
% am = A binary n-by-n matrix {0;1} representing the adjcency matrix.

function am = causal_adjacency(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('r',@(x)validateattributes(x,{'double'},{'real' '2d' 'nonempty'}));
        ip.addOptional('sst',0.05,@(x)validateattributes(x,{'double'},{'real' 'finite' '>' 0 '<=' 0.1 'scalar'}));
        ip.addOptional('rp',false,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    r = validate_input(ipr.r);
    sst = ipr.sst;
    rp = ipr.rp;

    nargoutchk(1,1);

    am = causal_adjacency_internal(r,sst,rp);

end

function am = causal_adjacency_internal(r,sst,rp)

    up = isempty(getCurrentTask());

    n = size(r,2);

    nan_indices = any(isnan(r),1);
    nok = sum(~nan_indices);

    seq = (1:n).';
    seq(nan_indices) = [];

    i = repelem(seq,nok,1);
    j = repmat(seq,nok,1); 

    indices = i == j;
    i(indices) = [];
    j(indices) = [];

    r_in = arrayfun(@(x)r(:,x),i,'UniformOutput',false);
    r_out = arrayfun(@(x)r(:,x),j,'UniformOutput',false);

    k = nok^2 - nok;
    pvals = zeros(k,1);

    if (rp)
        if (up)
            parfor y = 1:k
                [~,pvals(y)] = linear_granger_causality(r_in{y},r_out{y});
            end
        else
            for y = 1:k
                [~,pvals(y)] = linear_granger_causality(r_in{y},r_out{y});
            end
        end
    else
        if (up)
            parfor y = 1:k
                [pvals(y),~] = linear_granger_causality(r_in{y},r_out{y});
            end
        else
            for y = 1:k
                [pvals(y),~] = linear_granger_causality(r_in{y},r_out{y});
            end
        end
    end

    am = zeros(n);
    am(sub2ind([n n],i,j)) = pvals < sst;

end

function [b,c,r] = hac_regression(y,x,ratio)

    t = length(y);

    [b,~,r] = regress(y,x);

    h = diag(r) * x;
    q_hat = (x.' * x) / t;
    o_hat = (h.' * h) / t;

    l = round(ratio * t,0);

    for i = 1:(l - 1)
        o_tmp = (h(1:(t-i),:).' * h((1+i):t,:)) / (t - i);
        o_hat = o_hat + (((l - i) / l) * (o_tmp + o_tmp.'));
    end

    c = linsolve(q_hat,o_hat) / q_hat;

end

function [pval,pval_robust] = linear_granger_causality(in,out)

    t = length(in);
    y = out(2:t,1);
    x = [out(1:t-1) in(1:t-1)];

    [b,c,r] = hac_regression(y,x,0.1);

    xxi = inv(x.' * x);
    s2 = (r.' * r) / (t - 3);
    t_coefficients = b(2) / sqrt(s2 * xxi(2,2));

    pval = 1 - normcdf(t_coefficients);
    pval_robust = 1 - normcdf(b(2) / sqrt(c(2,2) / (t - 1)));

end

function r = validate_input(r)

    [t,n] = size(r);

    if ((t < 5) || (n < 2))
        error('The value of ''r'' is invalid. Expected input to be a matrix with a minimum size of 5x2.');
    end

end
