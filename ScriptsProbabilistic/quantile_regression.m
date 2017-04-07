% [INPUT]
% y    = A numeric vector representing the dependant variable.
% x    = A numeric t-by-n matrix with each column representing an independant variable.
% k    = A float representing the sample quantile.
%
% [OUTPUT]
% beta = A vector of floats containing the estimated regression coefficients.
% serr = A vector of floats containing the standard errors.
% tcoe = A vector of floats containing the t-Student coefficients.
% pval = A vector of floats containing the p-values.

function [beta,serr,tcoe,pval] = quantile_regression(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('y',@(x)validateattributes(x,{'numeric'},{'vector','finite','nonempty','nonnan','real'}));
        ip.addRequired('x',@(x)validateattributes(x,{'numeric'},{'2d','finite','nonempty','nonnan','real'}));
        ip.addRequired('k',@(x)validateattributes(x,{'double','single'},{'scalar','real','finite','>',0,'<',1}));
    end

    ip.parse(varargin{:});
    ip_res = ip.Results;

    [beta,serr,tcoe,pval] = quantile_regression_internal(ip_res.y,ip_res.x,ip_res.k);

end

function [beta,serr,tcoe,pval] = quantile_regression_internal(y,x,k)

    t = length(y);
    [xn,xm] = size(x);

    x = [ones(xn,1) x];
    xm = xm + 1;
    xs = x;

    beta = ones(xm,1);

    diff = 1;
    iter = 0;

    while ((diff > 1e-6) && (iter < 1000))
        xst = xs';
        beta_0 = beta;

        beta = ((xst * x) \ xst) * y;

        rsd = y - (x * beta);
        rsd(abs(rsd)<0.000001) = 0.000001;
        rsd(rsd<0) = k * rsd(rsd<0);
        rsd(rsd>0) = (1 - k) * rsd(rsd>0);
        rsd = abs(rsd);

        z = zeros(xn,xm);

        for i = 1:xm 
            z(:,i) = x(:,i) ./ rsd;
        end

        xs = z;
        beta_1 = beta;
        
        diff = max(abs(beta_1 - beta_0));
        iter = iter + 1;
    end

    if (nargout > 1)
        e = y - (x * beta);
        iqre = iqr(e);

        if (k == 0.5)
            h = (0.9 * std(e)) / (t ^ 0.2);
        else
            h = (0.9 * min(std(e),(iqre / 1.34))) / (t ^ 0.2);
        end

        u = exp(-(e / h));
        f_hat = (1 / (t * h)) * sum(u ./ ((1 + u) .^ 2));

        d(t,t) = 0;
        dgn = diag(d);
        dgn(e<=0) = ((1 - k) / f_hat) ^ 2;
        dgn(e>0) = (k / f_hat) ^ 2;
        d = diag(dgn);

        xt = x';

        serr = diag(((xt * x) ^ -1) * xt * d * x * ((xt * x) ^ -1)) .^ 0.5;
        
        if (nargout > 2)
            tcoe = beta ./ serr;
        end
        
        if (nargout > 3)
            pval = 2 * (1 - tcdf(abs(tcoe),(t - xm)));
        end
    end

end
