% [INPUT]
% data = A float t-by-n matrix {~} representing the model input, defined as follows:
%   - for Baseline MEM and Spline MEM:
%     * first column: endogenous variable [0,Inf), mandatory;
%     * other columns: exogenous variables [0,Inf), optional;
%   - for Asymmetric MEM and Asymmetric Power MEM:
%     * first column: endogenous variable [0,Inf), mandatory;
%     * second column: logarithmic returns (-Inf,Inf), mandatory;
%     * other columns: exogenous variables [0,Inf), optional.
% type = A string representing the model type:
%   - 'B' for Baseline MEM;
%   - 'A' for Asymmetric MEM;
%   - 'P' for Asymmetric Power MEM;
%   - 'S' for Spline MEM.
% q = An integer [1,Inf) representing the first order of the model (optional, default=1).
% p = An integer [1,Inf) representing the second order of the model (optional, default=1).
%
% [OUTPUT]
% m = A column vector of floats (-Inf,Inf) of length t representing the conditional means.
% e = A column vector of floats (-Inf,Inf) of length t representing the Cox-Snell residuals.
% mem_params = A vector of floats (-Inf,Inf) representing the MEM parameters.
% dist_params = A vector of floats (-Inf,Inf) representing the distribution parameters.

function [mu,e,mem_params,dist_params] = multiplicative_error(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data',@(x)validateattributes(x,{'double'},{'real' 'finite' '2d' 'nonempty'}));
        ip.addRequired('type',@(x)any(validatestring(x,{'A' 'B' 'P' 'S'})));
        ip.addOptional('q',1,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 1 'scalar'}));
        ip.addOptional('p',1,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 1 'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    [v,d,z,type] = validate_input(ipr.data,ipr.type);
    q = ipr.q;
    p = ipr.p;

    nargoutchk(1,4);

    [mu,e,mem_params,dist_params] = multiplicative_error_internal(v,d,z,type,q,p);

end

function [mu,e,mem_params,dist_params] = multiplicative_error_internal(v,d,z,type,q,p)

    persistent options;

    if (isempty(options))
        options = optimset(optimset(@fmincon),'Algorithm','sqp','Diagnostics','off','Display','off','LargeScale','off','MaxSQPIter',1000,'TolFun',1e-6);
    end

    switch (type)
        case 'B'
            [params,mu,e] = multiplicative_error_baseline(v,z,q,p,options);
        case 'A'
            [params,mu,e] = multiplicative_error_asymmetric(v,d,z,q,p,options);
        case 'P'
            [params,mu,e] = multiplicative_error_asymmetric_power(v,d,z,q,p,options);
        otherwise
            [params,mu,e] = multiplicative_error_spline(v,z,q,p,options);
    end

    mem_params = params(1:end-1);

    gamma = params(end);
    kappa = 1 / gamma;
    lambda = exp(gammaln(kappa) - gammaln(kappa + (1 / gamma)));
    dist_params = [gamma; kappa; lambda];

    e = -log(1 - gamcdf((e ./ lambda).^gamma,kappa,1));

end

function [params,mu,e] = multiplicative_error_baseline(v,z,q,p,options)

    vn = numel(v);
    vm = mean(v);

    zn = size(z,2);

    offset = max([zn q p]);

    qp = q + p;
    qpz = qp + zn;

    tol = 2 * options.TolCon;

    x0 = [(ones(q,1) .* (0.10 / q)); (ones(p,1) .* (0.75 / p)); (ones(zn,1) .* (0.01 / zn)); 0.5];
    ai = [[-eye(qpz) zeros(qpz,1)]; [ones(1,qpz) 0]];
    bi = [(ones(qpz,1) .* -tol); (1 - tol)];
    lb = ones(qpz + 1,1) .* tol;
    ub = [Inf(q,1); ones(p,1); Inf(zn + 1,1);];

    params = fmincon(@(x)likelihood(x,v,z,vn,vm,zn,q,p,offset,qp,qpz),x0,ai,bi,[],[],lb,ub,[],options);
    [~,mu,e] = likelihood(params,v,z,vn,vm,zn,q,p,offset,qp,qpz);

    function [ll,mu,e] = likelihood(x,v,z,vn,vm,zn,q,p,offset,qp,qpz)

        a = x(1:q);
        b = x(q+1:qp);
        y = x(qp+1:qpz);
        o = 1 - sum(a) - sum(b) - sum(y);

        mu = zeros(vn,1);
        mu(1:offset) = vm;

        for i = offset+1:vn
            mu_i = o;

            for j = 1:q
                mu_i = mu_i + (a(j) * v(i - j));
            end

            for j = 1:p
                mu_i = mu_i + (b(j) * mu(i - j));
            end

            for j = 1:zn
                mu_i = mu_i + (y(j) * z(i - j,j));
            end

            mu(i) = mu_i;
        end

        e = v ./ mu;

        ll = generalized_gamma_likelihood(v,mu,x(end));

    end

end

function [params,mu,e] = multiplicative_error_asymmetric(v,d,z,q,p,options)

    vn = numel(v);
    vm = mean(v);

    zn = size(z,2);

    offset = max([zn q p]);

    q2 = q * 2;
    q2p = q2 + p;
    q2pz = q2p + zn;

    tol = 2 * options.TolCon;

    x0 = [(ones(q,1) .* (0.05 / q)); (ones(q,1) .* (0.10 / q)); (ones(p,1) .* (0.70 / p)); (ones(zn,1) .* (0.01 / zn)); 0.5];
    ai = [[-eye(q2pz) zeros(q2pz,1)]; [ones(1,q) (ones(1,q) .* 0.5) ones(1,p) ones(1,zn) 0]];
    bi = [(ones(q2pz,1) .* -tol); (1 - tol)];
    lb = ones(q2pz + 1,1) .* tol;
    ub = [Inf(q2,1); ones(p,1); Inf(zn + 1,1);];

    params = fmincon(@(x)likelihood(x,v,d,z,vn,vm,zn,q,p,offset,q2,q2p,q2pz),x0,ai,bi,[],[],lb,ub,[],options);
    [~,mu,e] = likelihood(params,v,d,z,vn,vm,zn,q,p,offset,q2,q2p,q2pz);

    function [ll,mu,e] = likelihood(x,v,d,z,vn,vm,zn,q,p,offset,q2,q2p,q2pz)

        a = x(1:q);
        g = x(q+1:q2);
        b = x(q2+1:q2p);
        y = x(q2p+1:q2pz);
        o = 1 - sum(a) - (0.5 * sum(g)) - sum(b) - sum(y);

        mu = zeros(vn,1);
        mu(1:offset) = vm;

        for i = offset+1:vn
            mu_i = o;

            for j = 1:q
                mu_i = mu_i + ((a(j) + (g(j) * d(i - j))) * v(i - j));
            end

            for j = 1:p
                mu_i = mu_i + (b(j) * mu(i - j));
            end

            for j = 1:zn
                mu_i = mu_i + (y(j) * z(i - j,j));
            end

            mu(i) = mu_i;
        end

        e = v ./ mu;

        ll = generalized_gamma_likelihood(v,mu,x(end));

    end

end

function [params,mu,e] = multiplicative_error_asymmetric_power(v,d,z,q,p,options)

    vn = numel(v);
    vm = mean(v);

    zn = size(z,2);

    offset = max([zn q p]);

    q2 = q * 2;
    q2p = q2 + p;
    q2pz = q2p + zn + 1;
    pz = p + zn + 1;

    tol = 2 * options.TolCon;

    x0 = [(ones(q,1) .* (0.05 / q)); zeros(q,1); (ones(p,1) .* (0.75 / p)); 0.3; (ones(zn,1) .* (0.01 / zn)); 0.5];
    ai = [[-1 zeros(1,q2pz)]; [zeros(pz,2) -eye(pz) zeros(pz,1)]; [ones(1,q) zeros(1,q) ones(1,p) 0 ones(1,zn) 0]];
    bi = [(ones(q,1) .* -tol); (ones(pz,1) .* -tol); (1 - tol)];
    lb = [(ones(q,1) .* tol); (ones(q,1) .* -1); (ones(pz + 1,1) .* tol)];
    ub = [Inf(q,1); ones(q,1); ones(p,1); Inf(zn + 2,1);];

    params = fmincon(@(x)likelihood(x,v,d,z,vn,vm,zn,q,p,offset,q2,q2p,q2pz),x0,ai,bi,[],[],lb,ub,[],options);
    [~,mu,e] = likelihood(params,v,d,z,vn,vm,zn,q,p,offset,q2,q2p,q2pz);

    function [ll,mu,e] = likelihood(x,v,d,z,vn,vm,zn,q,p,offset,q2,q2p,q2pz)

        a = x(1:q);
        g = x(q+1:q2);
        b = x(q2+1:q2p);
        f = x(q2p+1);
        y = x(q2p+2:q2pz);
        o = 1 - sum(a) - sum(b) - sum(y);

        mu = zeros(vn,1);
        mu(1:offset) = vm;

        for i = offset+1:vn
            mu_i = o;

            for j = 1:q
                mu_i = mu_i + (a(j) * (abs(v(i - j)) - (g * v(i - j) * d(i - j)))^f);
            end

            for j = 1:p
                mu_i = mu_i + (b(j) * mu(i - j)^f);
            end

            for j = 1:zn
                mu_i = mu_i + (y(j) * z(i - j,j));
            end

            mu(i) = mu_i;
        end

        e = v ./ mu;

        ll = generalized_gamma_likelihood(v,mu,x(end));

    end

end

function [params,mu,e] = multiplicative_error_spline(v,z,q,p,options)

    vn = numel(v);
    vm = mean(v);

    zn = size(z,2);

    offset = max(q,p);

    qp = q + p;

    fh = @(x,k,qpk,qpkz)likelihood(x,v,z,vn,vm,zn,k,q,p,offset,qp,qpk,qpkz);
    tol = 2 * options.TolCon;

    knots = 2:15;
    knots_len = numel(knots);

    params_list = cell(knots_len,1);
    mu_list = cell(knots_len,1);
    e_list = cell(knots_len,1);

    results = [knots.' zeros(knots_len,1)];

    parfor k = knots
        qpk = qp + k + 2;
        qpkz = qpk + zn;
        kz = k + zn;

        x0 = [(ones(q,1) .* (0.10 / q)); (ones(p,1) .* (0.75 / p)); vm; zeros(kz + 1,1); 0.5];
        ai = [[-eye(qp) zeros(qp,kz + 3)]; [ones(1,qp) zeros(1,kz + 2) 0]];
        bi = [(ones(qp,1) .* -tol); (1 - tol)];
        lb = [(ones(qp,1) .* tol); zeros(kz + 2,1); tol];
        ub = [Inf(q,1); ones(p,1); Inf(kz + 3,1)];

        params = fmincon(@(x)fh(x,k,qpk,qpkz),x0,ai,bi,[],[],lb,ub,[],options);
        [~,mu,e,t] = fh(params,k,qpk,qpkz);

        params_list{k - 1} = params;
        mu_list{k - 1} = mu;
        e_list{k - 1} = e;

        results(k - 1,2) = sum(log(abs(mu - t))) + (log(vn) * ((k + 2) / vn));
    end

    results = sortrows(results,[2 1]);
    k = results(1,1);

    params = [k; params_list{k - 1}];
    mu = mu_list{k - 1};
    e = e_list{k - 1};

    function [ll,mu,e,t] = likelihood(x,v,z,vn,vm,zn,k,q,p,offset,qp,qpk,qpkz)

        a = x(1:q);
        b = x(q+1:qp);
        o = 1 - sum(a) - sum(b);

        c = x(qp+1);
        w = x(qp+2:qpk);
        y = x(qpk+1:qpkz);

        if (zn == 0)
            yz = [];
        else
            yz = repmat(y.',vn,1) .* z;
        end

        trend = (1:vn).';
        points = round((0:k-1) .* (vn / k),0);
        delta = repmat(trend,1,k) - repmat(points,vn,1);
        factors = delta > 0;

        t1 = (trend .* w(1)) ./ vn;
        t2 = repmat(w(2:end).',vn,1) .* factors .* (delta ./ vn).^2;
        t = c .* exp(sum([t1 t2 yz],2));

        mu = zeros(vn,1);
        mu(1:offset) = vm;

        for i = offset+1:vn
            mu_i = o;

            for j = 1:q
                mu_i = mu_i + (a(j) * (v(i - j)^2 / t(i - j)));
            end

            for j = 1:p
                mu_i = mu_i + (b(j) * mu(i - j));
            end

            mu(i) = mu_i;
        end

        mu = sqrt(mu .* t);
        e = v ./ mu;

        ll = generalized_gamma_likelihood(v,mu,x(end));

    end

end

function ll = generalized_gamma_likelihood(v,mu,gamma)

    kappa = 1 / gamma;
    lambda = exp(gammaln(kappa) - gammaln(kappa + (1 / gamma)));
    psi = kappa * gamma;

    ll_v = log(gamma) - ((1 + psi) * gammaln(kappa)) + (psi * gammaln(kappa + (1 / gamma))) + ((psi - 1) .* log(v)) - ((v ./ mu) ./ lambda).^gamma - (psi .* log(mu));

    if (any(~isfinite(ll_v)))
        ll = Inf;
    else
        ll = -sum(ll_v);
    end

end

function [v,d,z,type] = validate_input(data,type)

    n = size(data,2);

    v = data(:,1);

    if (any(v <= 0))
        error('The value of ''data'' is invalid. Expected input to contain only positive values in the first column.');
    end

    if (any(strcmp(type,{'A' 'P'})))
        if (n < 2)
            error('The value of ''data'' is invalid. Expected input to be a matrix with at least 2 columns.');
        end

        if (strcmp(type,'A'))
            d = data(:,2) < 0;
        else
            d = sign(data);
        end

        zn = n - 2;
        zo = 3;
    else
        d = [];

        zn = n - 1;
        zo = 2;
    end

    if (zn == 0)
        z = [];
    else
        z = data(:,zo:end);

        for i = 1:zn
            z_i = z(:,1);

            if (any(z_i <= 0))
                z(:,1) = z_i + min(z_i) + 1e-6;
            end
        end
    end

end
