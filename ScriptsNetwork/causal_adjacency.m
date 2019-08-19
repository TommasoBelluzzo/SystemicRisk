% [INPUT]
% data             = A numeric t-by-n matrix containing the network data.
% significance     = A float [0.00,0.20] representing the statistical significance threshold for the linear Granger-causality test (optional, default=0.05).
% robust           = A boolean indicating whether to use robust p-values (optional, default=true).
%
% [OUTPUT]
% adjacency_matrix = An numeric n-by-n matrix representing the adjacency matrix of the network.

function adjacency_matrix = causal_adjacency(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data',@(x)validateattributes(x,{'numeric'},{'2d','nonempty','real','finite'}));
        ip.addOptional('significance',0.05,@(x)validateattributes(x,{'double','single'},{'scalar','real','finite','>',0,'<=',0.20}));
        ip.addOptional('robust',true,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});
    ipr = ip.Results;
    
    adjacency_matrix = causal_adjacency_internal(ipr.data,ipr.significance,ipr.robust);

end

function adjacency_matrix = causal_adjacency_internal(data,significance,robust)

    n = size(data,2);
    i = repelem((1:n).',n,1);
    j = repmat((1:n).',n,1); 

    indices = i == j;
    i(indices) = [];
    j(indices) = [];

    d_in = arrayfun(@(x)data(:,x),i,'UniformOutput',false);
    d_out = arrayfun(@(x)data(:,x),j,'UniformOutput',false);
    
    k = n^2 - n;
    result = zeros(k,1);

    parfor y = 1:k
        in = d_in{y};
        out = d_out{y};
        
        if (robust)
            [pval,~] = linear_granger_causality(in,out);
        else
            [~,pval] = linear_granger_causality(in,out);
        end
        
        if (pval < significance)
            result(y) = 1;
        end
    end

    indices = sub2ind([n n],i,j);

    adjacency_matrix = zeros(n);
    adjacency_matrix(indices) = result;

end

function [pval,pval_robust] = linear_granger_causality(in,out)
    
    t = length(in);
    y = out(2:t,1);
    x = [out(1:t-1) in(1:t-1)];

    [beta,covariance] = hac_regression(y,x,0.1);
    
    residuals = y - (x * beta);
    c = inv(x.' * x);
    s2 = (residuals.' * residuals) / (t - 3);
    t_coefficients = beta(2) / sqrt(s2 * c(2,2));
    
    pval = 1 - normcdf(t_coefficients);
    pval_robust = 1 - normcdf(beta(2) / sqrt(covariance(2,2) / (t - 1)));

end

function [beta,covariance] = hac_regression(y,x,rat)

    t = length(y);

    beta = regress(y,x);
    residuals = y - (x * beta);

    h = diag(residuals) * x;
    l = round(rat * t);
    q_hat = (x.' * x) / t;
    o_hat = (h.' * h) / t;

    for i = 1:l-1
        o_tmp = 0;

        for j = 1:t-i
            o_tmp = o_tmp + h(j,:).' * h(j+i,:);
        end

        o_tmp = o_tmp / (t - i);
        o_hat = o_hat + (((l - i) / l) * (o_tmp + o_tmp.'));
    end

    covariance = (q_hat \ o_hat) / q_hat;

end
