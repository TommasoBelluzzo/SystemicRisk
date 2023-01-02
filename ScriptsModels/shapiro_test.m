% [INPUT]
% y = A vector of floats (-Inf,Inf) of length t representing the residuals.
% a = A float (0,1) representing the significance level (optional, default=0.05).
%
% [OUTPUT]
% h0 = A boolean representing the null hypothesis (residuals are normally distributed) rejection outcome.
% pval = A float [0,1] representing the test p-value.
% stat = A float (-Inf,Inf) representing the normalized test statistic.
%
% [NOTES]
% When residuals are leptokurtic, the Shapiro-Francia test is performed.
% When residuals are platykurtic, the Shapiro-Wilk test is performed.

function [h0,pval,stat] = shapiro_test(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('y',@(x)validateattributes(x,{'double'},{'real' 'finite' 'vector' 'nonempty'}));
        ip.addOptional('a',0.05,@(x)validateattributes(x,{'double'},{'real' 'finite' '>' 0 '<' 1 'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    y = validate_input(ipr.y);
    a = ipr.a;

    nargoutchk(2,3);

    [h0,pval,stat] = shapiro_test_internal(y,a);

end

function [h0,pval,stat] = shapiro_test_internal(x,alpha)

    x = sort(x);

    if (kurtosis(x) > 3)
        [pval,stat] = shapiro_francia(x);
    else
        [pval,stat] = shapiro_wilks(x);
    end

    h0  = (alpha >= pval);

end

function [pval,stat] = shapiro_francia(x)

    n = numel(x);

    x0 = x - mean(x);
    m = norminv(((1:n).' - 0.375) ./ (n + 0.25));
    w = (1 / sqrt(m.' * m)) .* m;
    k = (w.' * x)^2 / (x0.' * x0);

    nu = log(n);
    u1 = log(nu) - nu;
    u2 = log(nu) + (2 / nu);

    mu = -1.27250 + (1.05210 * u1);
    sigma = 1.03080 - (0.26758 * u2);

    stat = (log(1 - k) - mu) / sigma;
    pval = 1 - normcdf(stat,0,1);

end

function [pval,stat] = shapiro_wilks(x)

    n = numel(x);

    x0 = x - mean(x);
    m = norminv(((1:n).' - 0.375) ./ (n + 0.25));
    c = (1 / sqrt(m.' * m)) .* m;
    u = 1 / sqrt(n);

    pc_1 = [-2.7060560  4.434685 -2.071190 -0.147981 0.221157 c(n)];
    pc_2 = [-3.5826330  5.682633 -1.752461 -0.293762 0.042981 c(n - 1)];
    pc_3 = [-0.0006714  0.025054 -0.399780  0.544000];
    pc_4 = [-0.0020322  0.062767 -0.778570  1.382200];
    pc_5 = [ 0.0038915 -0.083751 -0.310820 -1.586100];
    pc_6 = [ 0.0030302 -0.082676 -0.480300];
    pc_7 = [ 0.459 -2.273];

    w = zeros(n,1);
    w(n) = polyval(pc_1,u);
    w(1) = -w(n);

    if (n >= 6)
        off = 3:n-3+1;

        w(n-1) = polyval(pc_2,u);
        w(2) = -w(n-1);

        phi = ((m.' * m) - (2 * m(n)^2) - (2 * m(n-1)^2)) /  (1 - (2 * w(n)^2) - (2 * w(n-1)^2));
    else
        off = 2:n-2+1;

        if (n == 3)
            w(1) = 1 / sqrt(2);
            w(n) = -w(1);

            phi = 1;
        else
            phi = ((m' * m) - (2 * m(n)^2)) / (1 - (2 * w(n)^2));
        end
    end

    w(off) = m(off) ./ sqrt(phi);

    k = (w.' * x)^2 / (x0.' * x0);

    if (n == 3)
        mu = 0;
        sigma = 1;
        kn = 0;
    elseif ((n >= 4) && (n <= 11))
        mu = polyval(pc_3,n);
        sigma = exp(polyval(pc_4,n));    
        gamma = polyval(pc_7,n);
        kn = -log(gamma - log(1 - k));
    else
        ln = log(n);
        mu = polyval(pc_5,ln);
        sigma = exp(polyval(pc_6,ln));
        kn = log(1 - k);
    end

    stat = (kn - mu) / sigma;

    if (n == 3)
        pval = (6 / pi()) * (asin(sqrt(k)) - asin(sqrt(0.75)));
    else
        pval = 1 - normcdf(stat,0,1);
    end

end

function y = validate_input(y)

    y = y(:);
    t = numel(y);

    if (t < 5)
        error('The value of ''y'' is invalid. Expected input to be a vector containing at least 5 elements.');
    end

end
