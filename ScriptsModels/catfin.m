% [INPUT]
% r = A vector of floats (-Inf,Inf) of length t representing the logarithmic returns.
% a = A float [0.01,0.10] representing the quantile used to calculate all the values-at-risk.
% g = A float [0.75,0.99] representing the weighting factor used to calculate the non-parametric value-at-risk (optional, default=0.98).
% u = A float [0.01,0.10] representing the threshold used to calculate the GPD value-at-risk (optional, default=0.05).
%
% [OUTPUT]
% var_np = A float (-Inf,0] representing the non-parametric value-at-risk.
% var_gpd = A float (-Inf,0] representing the GPD value-at-risk.
% var_gev = A float (-Inf,0] representing the GEV value-at-risk.
% var_sged = A float (-Inf,0] representing the SGED value-at-risk.

function [var_np,var_gpd,var_gev,var_sged] = catfin(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('r',@(x)validateattributes(x,{'double'},{'real' 'finite' 'vector' 'nonempty'}));
        ip.addRequired('a',@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.01 '<=' 0.10 'scalar'}));
        ip.addOptional('g',0.98,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.50 '<=' 0.99 'scalar'}));
        ip.addOptional('u',0.05,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.01 '<=' 0.10 'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    r = validate_input(ipr.r);
    a = ipr.a;
    g = ipr.g;
    u = ipr.u;

    nargoutchk(4,4);

    [var_np,var_gpd,var_gev,var_sged] = catfin_internal(r,a,g,u);

end

function [var_np,var_gpd,var_gev,var_sged] = catfin_internal(r,a,g,u)

    persistent options;

    if (isempty(options))
        options = optimset(optimset(@fsolve),'Diagnostics','off','Display','off');
    end

    t = numel(r);

    w = fliplr(((1 - g) / (1 - g^t)) .* (g .^ (0:1:t-1))).';  
    h = sortrows([r w],1);
    csw = cumsum(h(:,2));
    cswa = find(csw >= a);
    var_np = h(cswa(1),1);  

    k = round(t / (t * u),0);
    x_neg = -r; 
    x_neg_sorted = sort(x_neg);
    threshold = x_neg_sorted(t - k);
    excess = x_neg(x_neg > threshold) - threshold;
    gpd_params = gpfit(excess);
    [xi,beta,zeta] = deal(gpd_params(1),gpd_params(2),k / t);
    var_gpd = -(threshold + (beta / xi) * ((((1 / zeta) * a) ^ -xi) - 1));

    k = max(round(nthroot(t,1.81),0),5);
    block_maxima = find_block_maxima(r,t,k);
    theta = find_extremal_index(r,t,k);
    gev_params = gevfit(block_maxima);
    [xi,sigma,mu] = deal(gev_params(1),gev_params(2),gev_params(3));
    var_gev = -(mu - (sigma / xi) * (1 - (-(t / k) * theta * log(1 - a))^-xi));

    try
        sged_params = mle(r,'PDF',@sgedpdf,'Start',[mean(r) std(r) 0 1],'LowerBound',[-Inf 0 -1 0],'UpperBound',[Inf Inf 1 Inf]);
        [mu,sigma,lambda,kappa] = deal(sged_params(1),sged_params(2),sged_params(3),sged_params(4));
        var_sged = fsolve(@(x)sgedcdf(x,mu,sigma,lambda,kappa)-a,0,options);
    catch
        var_sged = NaN;
    end

    vars = min(0,[var_np var_gpd var_gev var_sged]);
    [var_np,var_gpd,var_gev,var_sged] = deal(vars(1),vars(2),vars(3),vars(4));

end

function block_maxima = find_block_maxima(x,t,k)

    c = floor(t / k);

    block_maxima = zeros(k,1);
    i = 1;

    for j = 1:k-1
        block_maxima(j) = max(x(i:i+c-1));
        i = i + c;
    end

    block_maxima(k) = max(x(i:end));

end

function theta = find_extremal_index(x,t,k)

    c = t - k + 1;
    y = zeros(c,1);

    for i = 1:c
        y(i,1) = (1 / t) * sum(x <= max(x(i:i+k-1)));
    end

    theta = ((1 / c) * sum(-k * log(y)))^-1;

end

function p = sgedcdf(x,mu,sigma,lambda,kappa)

    [t,n] = size(x);
    p = NaN(t,n);

    for i = 1:t
        for j = 1:n
            p(i) = integral(@(x)sgedpdf(x,mu,sigma,lambda,kappa),-Inf,x(i,j));
        end
    end

end

function y = sgedpdf(x,mu,sigma,lambda,kappa)

    g1 = gammaln(1 / kappa);
    g2 = gammaln(2 / kappa);
    g3 = gammaln(3 / kappa);

    a = exp(g2 - (0.5 * g1) - (0.5 * g3));
    s = sqrt(1 + (3 * lambda^2) - (4 * a^2 * lambda^2));

    theta = exp((0.5 * g1) - (0.5 * g3)) / s;
    delta = (2 * lambda * a) / s;

    c = exp(log(kappa) - (log(2 * sigma * theta) + g1));
    u = x - mu + (delta * sigma);

    y = c .* exp((-abs(u) .^ kappa) ./ ((1 + (sign(u) .* lambda)) .^ kappa) ./ theta^kappa ./ sigma^kappa); 

end

function r = validate_input(r)

    r = r(:);
    t = numel(r);

    if (t < 5)
        error('The value of ''r'' is invalid. Expected input to be a vector containing at least 5 elements.');
    end

end
