% [INPUT]
% r = A float t-by-n matrix (-Inf,Inf) representing the logarithmic returns.
% cds = A vector of floats [0,Inf) of length n representing the credit default swap spreads.
% lb = A vector of floats [0,Inf) of length n representing the liabilities.
% f = An integer [2,n], where n is the number of firms, representing the number of systematic risk factors (optional, default=2).
% lgd = A float (0,1] representing the loss given default, complement to recovery rate (optional, default=0.55).
% l = A float [0.05,0.20] representing the importance sampling threshold (optional, default=0.10).
% c = An integer [50,1000] representing the number of simulated samples (optional, default=100).
% it = An integer [5,100] representing the number of iterations to perform (optional, default=5).
%
% [OUTPUT]
% dip = A float [0,Inf) representing the Distress Insurance Premium.

function dip = distress_insurance_premium(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('r',@(x)validateattributes(x,{'double'},{'real' '2d' 'nonempty'}));
        ip.addRequired('cds',@(x)validateattributes(x,{'double'},{'real' 'nonnegative' 'vector' 'nonempty'}));
        ip.addRequired('lb',@(x)validateattributes(x,{'double'},{'real' 'nonnegative' 'vector' 'nonempty'}));
        ip.addOptional('f',2,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 2 'scalar'}));
        ip.addOptional('lgd',0.55,@(x)validateattributes(x,{'double'},{'real' 'finite' '>' 0 '<=' 1 'scalar'}));
        ip.addOptional('l',0.10,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.05 '<=' 0.20 'scalar'}));
        ip.addOptional('c',100,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 50 '<=' 1000 'scalar'}));
        ip.addOptional('it',5,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 5 '<=' 100 'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    [n,indices,r,cds,lb,f] = validate_input(ipr.r,ipr.cds,ipr.lb,ipr.f);
    lgd = ipr.lgd;
    l = ipr.l;
    c = ipr.c;
    it = ipr.it;

    nargoutchk(1,1);

    dip = distress_insurance_premium_internal(n,indices,r,cds,lb,f,lgd,l,c,it);

end

function dip = distress_insurance_premium_internal(n,indices,r,cds,lb,f,lgd,l,c,it)

    up = isempty(getCurrentTask());

    b = estimate_factor_loadings(r(:,indices),f);
    [dt,dw,ead,ead_volume,lgd] = estimate_default_parameters(cds(indices),lb(indices),n,lgd);

    bi = floor(c * 0.2);
    c2 = c^2;

    a = zeros(5,1);

    if (up)
        parfor i = 1:it 
            mcmc_p = slicesample(rand(1,f),c,'PDF',@(x)zpdf(x,dt,ead,lgd,b,l),'Thin',3,'BurnIn',bi);
            [mu,sigma,weights] = gmm_fit(mcmc_p,2);  
            [z,g] = gmm_evaluate(mu,sigma,weights,c);

            phi = normcdf((repmat(dt.',c,1) - (z * b.')) ./ (1 - repmat(sum(b.^2,2).',c,1)).^0.5);
            [theta,theta_p] = exponential_twist(phi,dw,l);

            losses = sum(repelem(dw.',c2,1) .* ((repelem(theta_p,c,1) >= rand(c2,n)) == 1),2);
            psi = sum(log((phi .* exp(repmat(theta,1,n) .* repmat(dw.',c,1))) + (1 - phi)),2);

            lr_z = repelem(mvnpdf(z) ./ g,c,1);
            lr_e = exp(-(repelem(theta,c,1) .* losses) + repelem(psi,c,1));
            lr = lr_z .* lr_e;

            a(i) = mean((losses > l) .* lr);
        end
    else
        for i = 1:it
            mcmc_p = slicesample(rand(1,f),c,'PDF',@(x)zpdf(x,dt,ead,lgd,b,l),'Thin',3,'BurnIn',bi);
            [mu,sigma,weights] = gmm_fit(mcmc_p,2);  
            [z,g] = gmm_evaluate(mu,sigma,weights,c);

            phi = normcdf((repmat(dt.',c,1) - (z * b.')) ./ (1 - repmat(sum(b.^2,2).',c,1)).^0.5);
            [theta,theta_p] = exponential_twist(phi,dw,l);

            losses = sum(repelem(dw.',c2,1) .* ((repelem(theta_p,c,1) >= rand(c2,n)) == 1),2);
            psi = sum(log((phi .* exp(repmat(theta,1,n) .* repmat(dw.',c,1))) + (1 - phi)),2);

            lr_z = repelem(mvnpdf(z) ./ g,c,1);
            lr_e = exp(-(repelem(theta,c,1) .* losses) + repelem(psi,c,1));
            lr = lr_z .* lr_e;

            a(i) = mean((losses > l) .* lr);
        end
    end

    dip = max(mean(a) * ead_volume,0);

end

function b = estimate_factor_loadings(r,f)

    rho = corr(r);
    f0 = eye(size(rho,1)) * 0.2;

    count = 0;
    error = 0.8;

    while ((count < 100) && (error > 0.01))
        [v,d] = eig(rho - f0,'vector');

        [~,sort_indices] = sort(d,'descend');
        sort_indices = sort_indices(1:f);

        d = diag(d(sort_indices));
        v = v(:,sort_indices);
        b = v * sqrt(d);

        f1 = diag(1 - diag(b * b.'));
        delta = f1 - f0;

        f0 = f1;

        count = count + 1;
        error = trace(delta * delta.');
    end

end

function [dt,dw,ead,ead_volume,lgd] = estimate_default_parameters(cds,liabilities,n,lgd)

    dt = norminv(1 - exp(-cds ./ lgd)).';

    liabilities_sum = sum(liabilities);
    ead = (liabilities / sum(liabilities_sum)).';
    ead_volume = liabilities_sum;

    if (lgd > 0.5)
        lgd = mean(cumsum(randtri((2 * lgd) - 1,lgd,1,[n 1000])),2);
    else
        lgd = mean(cumsum(randtri(0,lgd,1,[n 1000])),2);
    end

    dw = ead .* lgd;

end

function [theta,theta_p] = exponential_twist(phi,w,l)

    persistent options;

    if (isempty(options))
        options = optimset(optimset(@fminunc),'Diagnostics','off','Display','off','LargeScale','off');
    end

    [c,n] = size(phi);

    theta = zeros(c,1);
    theta_p = phi;

    w = [w zeros(n,1)];

    for i = 1:c
        phi_i = phi(i,:).';
        p = [phi_i (1 - phi_i)];

        threshold = sum(sum(w .* p,2),1);

        if (l > threshold)
            if (i == 1)
                x0 = 0;
            else
                x0 = theta(i-1);
            end

            [t,~,ef] = fminunc(@(x)sum(log(sum(p .* exp(w .* x),2)),1) - (x * l),x0,options);

            if (ef > 0)
                theta(i) = t;

                twist = p .* exp(w .* t(end));
                theta_p(i,:) = twist(:,1) ./ sum(twist,2);
            end
        end
    end

end

function [z,g] = gmm_evaluate(mu,sigma,weights,c)

    indices = datasample(1:numel(weights),c,'Replace',true,'Weights',weights);
    z = mvnrnd(mu(indices,:),sigma(:,:,indices),c);

    g = zeros(c,1);

    for i = 1:c
        g(i) = sum(mvnpdf(z(i,:),mu,sigma) .* weights);
    end

end

function [mu,sigma,weights] = gmm_fit(x,gm)

    [c,s] = size(x);

    m = x(randsample(c,gm),:);
    [~,indices] = max((x * m.') - repmat(dot(m,m,2).' / 2,c,1),[],2);
    [u,~,indices] = unique(indices);

    while (numel(u) ~= gm)
        m = x(randsample(c,gm),:);
        [~,indices] = max((x * m.') - repmat(dot(m,m,2).' / 2,c,1),[],2);
        [u,~,indices] = unique(indices);
    end

    r = zeros(c,gm);
    r(sub2ind([c gm],1:c,indices.')) = 1;

    [~,indices] = max(r,[],2);
    r = r(:,unique(indices));

    llh_old = -Inf;
    count = 1;
    converged = false;

    while ((count < 10000) && ~converged)
        count = count + 1;

        rk = size(r,2);
        rs = sum(r,1).';
        rq = sqrt(r);

        mu = (r.' * x) .* repmat(1 ./ rs,1,s);
        sigma = zeros(s,s,rk);
        rho = zeros(c,rk);
        weights = rs ./ c;

        for j = 1:rk
            x0 = x - repmat(mu(j,:),c,1);

            o = x0 .* repmat(rq(:,j),1,s);
            h = ((o.' * o) ./ rs(j)) + (eye(s) .* 1e-6);
            sigma(:,:,j) = h;

            cu = chol(h,'upper');
            q0 = linsolve(cu.',x0.');
            q1 = dot(q0,q0,1);
            nc = (s * log(2 * pi())) + (2 * sum(log(diag(cu))));
            rho(:,j) = (-(nc + q1) / 2) + log(weights(j));
        end

        rho_max = max(rho,[],2);
        t = rho_max + log(sum(exp(rho - repmat(rho_max,1,rk)),2));
        fi = ~isfinite(rho_max);
        t(fi) = rho_max(fi);
        llh = sum(t) / c;

        r = exp(rho - repmat(t,1,rk));

        [~,indices] = max(r,[],2);
        u = unique(indices);

        if (size(r,2) ~= numel(u))
            r = r(:,u);
        else
            converged = (llh - llh_old) < (1e-8 * abs(llh));
        end

        llh_old = llh;
    end

end

function r = randtri(a,b,c,size)

    d = (b - a) / (c - a);

    p = rand(size);
    r = p;

    t = ((p >= 0) & (p <= d));
    r(t) = a + sqrt(p(t) * (b - a) * (c - a));

    t = ((p <= 1) & (p > d));
    r(t) = c - sqrt((1 - p(t)) * (c - b) * (c - a));

end

function p = zpdf(z,dt,ead,lgd,b,l) 

    p0 = normcdf((dt - (b * z.')) ./ (1 - sum(b.^2,2)).^0.5);
    mu = sum(ead .* lgd .* p0);
    sigma = sqrt((ead.' .^ 2) * sum((-1 .* lgd).^2 .* p0 .* (1 - p0),2));

    p = max(1e-16,(1 - normcdf((l - mu) / sigma)) * mvnpdf(z));

end

function [n,indices,r,cds,lb,f] = validate_input(r,cds,lb,f)

    [t,n_tot] = size(r);

    if ((t < 5) || (n_tot < 2))
        error('The value of ''r'' is invalid. Expected input to be a matrix with a minimum size of 5x2.');
    end

    cds = cds(:).';

    if (numel(cds) ~= n_tot)
        error(['The value of ''cds'' is invalid. Expected input to be a vector of ' num2str(n_tot) ' elements.']);
    end

    if (all(cds >= 1))
        cds = cds ./ 10000;
    end

    lb = lb(:).';

    if (numel(lb) ~= n_tot)
        error(['The value of ''lb'' is invalid. Expected input to be a vector of ' num2str(n_tot) ' elements.']);
    end

    indices = (sum(isnan(r),1) == 0) & ~isnan(cds) & ~isnan(lb);
    n = sum(indices);

    if (n < 2)
        error('Input data must contain at least 2 valid time series without NaN values.');
    end

    f = min(f,n);

end
