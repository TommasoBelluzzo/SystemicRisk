% [INPUT]
% data_in  = A numeric vector containing the input network data.
% data_out = A numeric vector containing the output network data.
%
% [OUTPUT]
% pval     = A vector of floats containing the p-values.
% pval_rob = A vector of floats containing the robust p-values.

function [pval,pval_rob] = linear_granger_causality(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data_in',@(x)validateattributes(x,{'numeric'},{'vector','finite','nonempty','nonnan','real'}));
        ip.addRequired('data_out',@(x)validateattributes(x,{'numeric'},{'vector','finite','nonempty','nonnan','real'}));
    end

    ip.parse(varargin{:});
    ip_res = ip.Results;
    
    [pval,pval_rob] = linear_granger_causality_internal(ip_res.data_in,ip_res.data_out);

end

function [pval,pval_rob] = linear_granger_causality_internal(data_in,data_out)
    
    t = length(data_in);

    y = data_out(2:t,1);
    x = [data_out(1:t-1) data_in(1:t-1)];

    [beta,v_hat] = hac_regression(y,x,0.1);

    rsd = y - (x * beta);
    c = inv(x' * x);
    s2 = (rsd' * rsd) / (t - 3);
    tcoe = beta(2) / sqrt(s2 * c(2,2));
    
    pval = 1 - normcdf(tcoe);
    pval_rob = 1 - normcdf(beta(2) / sqrt(v_hat(2,2) / (t - 1)));

end