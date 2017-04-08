% [INPUT]
% y     = A numeric vector representing the dependant variable.
% x     = A numeric t-by-n matrix with each column representing an independant variable.
% rat   = A float representing the truncation lag to number of observations ratio used to construct the HAC estimators.

% [OUTPUT]
% beta  = A vector of floats containing the estimated regression coefficients.
% v_hat = A n-by-n matrix of floats containing the HAC estimators.

function [beta,v_hat] = hac_regression(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('y',@(x)validateattributes(x,{'numeric'},{'vector','finite','nonempty','nonnan','real'}));
        ip.addRequired('x',@(x)validateattributes(x,{'numeric'},{'2d','finite','nonempty','nonnan','real'}));
        ip.addRequired('rat',@(x)validateattributes(x,{'double','single'},{'scalar','real','finite','>',0,'<',1}));
    end

    ip.parse(varargin{:});
    ip_res = ip.Results;

    [beta,v_hat] = hac_regression_internal(ip_res.y,ip_res.x,ip_res.rat);

end

function [beta,v_hat] = hac_regression_internal(y,x,rat)

    t = length(y);

    beta = regress(y,x);
    rsd = y - (x * beta);

    h = diag(rsd) * x;
    l = round(rat * t);
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