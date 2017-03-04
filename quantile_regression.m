% [INPUT]
% y = A column vector representing the dependant variable.
% x = A t-by-n matrix representing the independant variables.
% k = A scalar [0,1] representing the sample quantile.
% [OUTPUT]
% beta   = The estimated regression coefficients.
% stderr = The standard errors.
% tcoeff = The t-Students.
% pval   = The p-values.

function [beta,stderr,tcoeff,pval] = quantile_regression(y,x,k)

    yn = length(y);
    [xn,xm] = size(x);

    x = [ones(xn,1) x];
    xm = xm + 1;
    xstar = x;

    beta = ones(xm,1);
    diff = 1; iter = 0;

    while ((iter < 1000) && (diff > 1e-6))
        xstart = xstar';
        
        b0 = beta;
        beta = ((xstart * x) \ xstart) * y;

        res = y - (x * beta);
        res(abs(res)<0.000001) = 0.000001;
        res(res<0) = k * res(res<0);
        res(res>0) = (1 - k) * res(res>0);
        res = abs(res);

        z=[];

        for i = 1:xm 
            z0 = x(:,i) ./ res;
            z = [z z0];
        end

        xstar = z;
        b1 = beta;
        
        diff = max(abs(b1 - b0));
        iter = iter + 1;
    end

    e = y - (x * beta);
    iqre = iqr(e);

    if (k == 0.5)
        h = (0.9 * std(e)) / (yn ^ 0.2);
    else
        h = (0.9 * min(std(e), (iqre / 1.34))) / (yn ^ 0.2);
    end

	u = e / h;
    uem = exp(-u);
    fhat0 = (1 / (yn * h)) * sum(uem ./ ((1 + uem) .^ 2));
    
    d(yn,yn) = 0;
    dgn = diag(d);
    dgn(e<=0) = ((1 - k) / fhat0) ^ 2;
    dgn(e>0) = (k / fhat0) ^ 2;
    d = diag(dgn);
    
    xt = x';
    vcq = (xt * x)^(-1) * xt * d * x * (xt * x)^(-1);
    dgnvcq = diag(vcq);

    stderr = dgnvcq .^ 0.5;
    tcoeff = beta ./ stderr;
    pval = 2 * (1 - tcdf(abs(tcoeff), (yn - xm)));

end