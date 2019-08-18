% [INPUT]
% data = A numeric t-by-n matrix containing the network data.
% sst  = A float [0.00,0.20] representing the statistical significance threshold for the linear Granger-causality test (optional, default=0.05).
% rob  = A boolean indicating whether to use robust p-values (optional, default=true).
%
% [OUTPUT]
% adjm = An numeric n-by-n matrix representing the adjacency matrix of the network.

function adjm = calculate_adjacency_matrix(varargin)

    persistent p;

    if (isempty(p))
        p = inputParser();
        p.addRequired('data',@(x)validateattributes(x,{'numeric'},{'2d','finite','nonempty','nonnan','real'}));
        p.addOptional('sst',0.05,@(x)validateattributes(x,{'double','single'},{'scalar','real','finite','>',0,'<=',0.20}));
        p.addOptional('rob',true,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    p.parse(varargin{:});
    res = p.Results;
    
    adjm = calculate_adjacency_matrix_internal(res.data,res.sst,res.rob);

end

function adjm = calculate_adjacency_matrix_internal(data,sst,rob)

    n = size(data,2);
    n_seq = (1:n).';

    i = repelem(n_seq,n,1);
    j = repmat(n_seq,n,1); 

    idx = i == j;
    i(idx) = [];
    j(idx) = [];

    din = arrayfun(@(x)data(:,x),i,'UniformOutput',false);
    dout = arrayfun(@(x)data(:,x),j,'UniformOutput',false);
    
    k = n^2 - n;
    res = zeros(k,1);

    parfor y = 1:k
        in = din{y};
        out = dout{y};
        
        if (rob)
            [pval,~] = linear_granger_causality(in,out);
        else
            [~,pval] = linear_granger_causality(in,out);
        end
        
        if (pval < sst)
            res(y) = 1;
        end
    end

    idx = sub2ind([n n],i,j);

    adjm = zeros(n);
    adjm(idx) = res;

end

function [pval,pval_rob] = linear_granger_causality(in,out)
    
    t = length(in);

    y = out(2:t,1);
    x = [out(1:t-1) in(1:t-1)];

    [beta,cov] = hac_regression(y,x,0.1);
    
    rsd = y - (x * beta);
    c = inv(x.' * x);
    s2 = (rsd.' * rsd) / (t - 3);
    tcoe = beta(2) / sqrt(s2 * c(2,2));
    
    pval = 1 - normcdf(tcoe);
    pval_rob = 1 - normcdf(beta(2) / sqrt(cov(2,2) / (t - 1)));

end

function [beta,cov] = hac_regression(y,x,rat)

    t = length(y);

    beta = regress(y,x);
    rsd = y - (x * beta);

    h = diag(rsd) * x;
    l = round(rat * t);
    q_hat = (x.' * x) / t;
    o_hat = (h.' * h) / t;

    for i = 1:l-1
        otmp = 0;

        for j = 1:t-i
            otmp = otmp + h(j,:).' * h(j+i,:);
        end

        otmp = otmp / (t - i);
        o_hat = o_hat + (((l - i) / l) * (otmp + otmp.'));
    end

    cov = (q_hat \ o_hat) / q_hat;

end
