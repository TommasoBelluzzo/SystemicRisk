% [INPUT]
% y     = A column vector representing the dependant variable.
% x     = A t-by-n matrix representing the independant variables.
% ratio = A scalar [0,1] representing the truncation lag to number of observations ratio used to construct the HAC estimators.

% [OUTPUT]
% beta  = A column vector containing the estimated regression coefficients.
% v_hat = An n-by-n matrix containing the HAC estimators.

function [beta,v_hat] = hac_regression(y,x,ratio)

    t = length(y);

    beta = regress(y,x);
    rsd = y - (x * beta);

    h = diag(rsd) * x;
    l = round(ratio * t);
    q_hat = (x' * x) / t;
    o_hat = (h' * h) / t;

    for i = 1:l-1
        otmp = 0;

        for j = 1:t-i
            otmp = otmp + h(j,:)' * h(j+i,:);
        end

        otmp = otmp / (t - i);
        o_hat = o_hat + (((l - i) / l) * (otmp + otmp'));
    end

    v_hat = (q_hat \ o_hat) / q_hat;

end