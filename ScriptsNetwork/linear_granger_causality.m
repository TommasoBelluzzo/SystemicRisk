% [INPUT]
% ret_in   = A column vector containing input firm log returns.
% ret_out  = A column vector containing output firms log returns.
%
% [OUTPUT]
% pval     = A column vector containing the p-values.
% pval_rob = A column vector containing the robust p-values.

function [pval,pval_rob] = linear_granger_causality(ret_in,ret_out)
    
    t = length(ret_in);

    y = ret_out(2:t,1);
    x = [ret_out(1:t-1) ret_in(1:t-1)];

    [beta,v_hat] = hac_regression(y,x,0.1);

    rsd = y - (x * beta);
    c = inv(x' * x);
    s2 = (rsd' * rsd) / (t - 3);
    tcoe = beta(2) / sqrt(s2 * c(2,2));
    
    pval = 1 - normcdf(tcoe);
    pval_rob = 1 - normcdf(beta(2) / sqrt(v_hat(2,2) / (t - 1)));

end