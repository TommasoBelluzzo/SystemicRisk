% [INPUT]
% r = A float t-by-2 matrix (-Inf,Inf) representing the logarithmic returns, in which:
%   - the first column represents the market returns;
%   - the second column represents the firm returns.
% cp = A vector of floats [0,Inf) of length t representing the market capitalization of the firm.
% lb = A vector of floats [0,Inf) of length t representing the liabilities of the firm.
% lbr = A vector of floats [0,Inf) of length t representing the forward-rolled liabilities of the firm.
% sv = A float t-by-k matrix (-Inf,Inf) representing the state variables.
% a = A float [0.01,0.10] representing the target quantile.
% d = A float [0.1,0.6] representing the crisis threshold for the market index decline used to calculate the LRMES (optional, default=0.4).
% car = A float [0.03,0.20] representing the capital adequacy ratio used to calculate SES and SRISK (optional, default=0.08).
%
% [OUTPUT]
% beta = A column vector of floats [0,Inf) of length t representing the CAPM Beta.
% var = A column vector of floats [0,Inf) of length t representing the Value-at-Risk.
% es = A column vector of floats [0,Inf) of length t representing the Expected Shortfall.
% covar = A column vector of floats [0,Inf) of length t representing the Conditional Value-at-Risk.
% dcovar = A column vector of floats [0,Inf) of length t representing the Delta Conditional Value-at-Risk.
% mes = A column vector of floats [0,Inf) of length t representing the Marginal Expected Shortfall.
% ses = A column vector of floats [0,Inf) of length t representing the Systemic Expected Shortfall.
% srisk = A column vector of floats [0,Inf) of length t representing the Conditional Capital Shortfall Index.

function [beta,var,es,covar,dcovar,mes,ses,srisk] = cross_sectional_metrics(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('r',@(x)validateattributes(x,{'double'},{'real' 'finite' '2d' 'nonempty' 'size' [NaN 2]}));
        ip.addRequired('cp',@(x)validateattributes(x,{'double'},{'real' 'finite' 'vector' 'nonempty'}));
        ip.addRequired('lb',@(x)validateattributes(x,{'double'},{'real' 'finite' 'vector' 'nonempty'}));
        ip.addRequired('lbr',@(x)validateattributes(x,{'double'},{'real' 'finite' 'vector' 'nonempty'}));
        ip.addRequired('sv',@(x)validateattributes(x,{'double'},{'real' 'finite'}));
        ip.addRequired('a',@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.01 '<=' 0.10 'scalar'}));
        ip.addOptional('d',0.4,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.1 '<=' 0.6 'scalar'}));
        ip.addOptional('car',0.08,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.03 '<=' 0.20 'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    [r,cp,lb,lbr,sv] = validate_input(ipr.r,ipr.cp,ipr.lb,ipr.lbr,ipr.sv);
    a = ipr.a;
    d = ipr.d;
    car = ipr.car;

    nargoutchk(8,8);

    [beta,var,es,covar,dcovar,mes,ses,srisk] = cross_sectional_metrics_internal(r,cp,lb,lbr,sv,a,d,car);

end

function [beta,var,es,covar,dcovar,mes,ses,srisk] = cross_sectional_metrics_internal(r,cp,lb,lbr,sv,a,d,car)

    rm = r(:,1);
    rm_0 = rm - mean(rm);

    rf = r(:,2);
    rf_0 = rf - mean(rf);

    [p,h] = dcc_gjrgarch(r);
    sm = sqrt(h(:,1));
    sf = sqrt(h(:,2));
    rho = squeeze(p(1,2,:));

    beta = rho .* (sf ./ sm);

    c = quantile((rf_0 ./ sf),a);
    var = -1 .* min(sf * c,0);
    es = -1 .* min(sf * -(normpdf(c) / a),0);

    [covar,dcovar] = calculate_covar(rm_0,rf_0,-var,sv,a);
    [mes,lrmes] = calculate_mes(rm_0,sm,rf_0,sf,rho,beta,a,d);
    ses = calculate_ses(cp,lb,car);
    srisk = calculate_srisk(cp,lbr,lrmes,car);

end

function [covar,dcovar] = calculate_covar(rm_0,rf_0,var,sv,a)

    if (isempty(sv))
        b = quantile_regression(rm_0,rf_0,a);
        covar = b(1) + (b(2) .* var);
    else
        b = quantile_regression(rm_0(2:end),[rf_0(2:end) sv(1:end-1,:)],a);
        covar = b(1) + (b(2) .* var(2:end));

        for i = 1:size(sv,2)
            covar = covar + (b(i+2) .* sv(1:end-1,i));
        end

        covar = [covar(1); covar];
    end

    dcovar = b(2) .* (var - repmat(median(rf_0),length(rm_0),1));

    covar = -1 .* min(covar,0);
    dcovar = -1 .* min(dcovar,0);

end

function [mes,lrmes] = calculate_mes(rm_0,sm,rf_0,sf,rho,beta,a,d)

    c = quantile(rm_0,a);
    z = sqrt(1 - rho.^2);

    u = rm_0 ./ sm;
    x = ((rf_0 ./ sf) - (rho .* u)) ./ z;

    r0_n = 4 / (3 * length(rm_0));
    r0_s = min([std(rm_0 ./ sm) (iqr(rm_0 ./ sm) ./ 1.349)]);
    h = r0_s * r0_n ^0.2;

    f = normcdf(((c ./ sm) - u) ./ h);
    f_sum = sum(f);

    k1 = sum(u .* f) ./ f_sum;
    k2 = sum(x .* f) ./ f_sum;

    mes = -1 .* min((sf .* rho .* k1) + (sf .* z .* k2),0);
    lrmes = 1 - exp(log(1 - d) .* beta);

end

function ses = calculate_ses(cp,lb,car)

    lb_pc = [0; diff(lb) ./ lb(1:end-1)];
    eq_pc = [0; diff(cp) ./ cp(1:end-1)];

    ses = (car .* lb .* (1 + lb_pc)) - ((1 - car) .* cp .* (1 + eq_pc));
    ses(ses < 0) = 0;

end

function srisk = calculate_srisk(cp,lb,lrmes,car)

    srisk = (car .* lb) - ((1 - car) .* (1 - lrmes) .* cp);
    srisk(srisk < 0) = 0;

end

function b = quantile_regression(y,x,a)

    [n,m] = size(x);
    m = m + 1;

    x = [ones(n,1) x];
    x_star = x;

    b = ones(m,1);

    diff = 1;
    i = 0;

    while ((diff > 1e-6) && (i < 1000))
        x_star_t = x_star.';
        b_0 = b;

        b = linsolve(x_star_t * x,x_star_t) * y;

        rsd = y - (x * b);
        rsd(abs(rsd) < 1e-06) = 1e-06;
        rsd(rsd < 0) = a * rsd(rsd < 0);
        rsd(rsd > 0) = (1 - a) * rsd(rsd > 0);
        rsd = abs(rsd);

        z = zeros(n,m);

        for j = 1:m 
            z(:,j) = x(:,j) ./ rsd;
        end

        x_star = z;
        b_1 = b;

        diff = max(abs(b_1 - b_0));
        i = i + 1;
    end

end

function [r,cp,lb,lbr,sv] = validate_input(r,cp,lb,lbr,sv)

    t = size(r,1);

    if (t < 5)
        error('The value of ''r'' is invalid. Expected input to be a matrix with at least 5 rows.');
    end

    if (~isvector(cp))
        error('The value of ''cp'' is invalid. Expected input to be a vector.');
    end

    if (numel(cp) ~= t)
        error(['The value of ''cp'' is invalid. Expected input to contain ' num2str(t) ' elements.']);
    end

    cp = cp(:);

    if (~isvector(lb))
        error('The value of ''lb'' is invalid. Expected input to be a vector.');
    end

    if (numel(lb) ~= t)
        error(['The value of ''lb'' is invalid. Expected input to contain ' num2str(t) ' elements.']);
    end

    lb = lb(:);

    if (~isvector(lbr))
        error('The value of ''lbr'' is invalid. Expected input to be a vector.');
    end

    if (numel(lbr) ~= t)
        error(['The value of ''lbr'' is invalid. Expected input to contain ' num2str(t) ' elements.']);
    end

    lbr = lbr(:);

    if (~isempty(sv))
        if (~ismatrix(sv))
            error('The value of ''sv'' is invalid. Expected input to be a matrix.');
        end

        if (size(sv,1) ~= t)
            error(['The value of ''lbr'' is invalid. Expected input to contain ' num2str(t) ' rows.']);
        end
    end

end
