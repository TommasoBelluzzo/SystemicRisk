% [INPUT]
% r = A float t-by-n matrix (-Inf,Inf) representing the logarithmic returns.
% pods = A vector of floats [0,1] of length n representing the probabilities of default.
% md = A string representing the multivariate distribution used by the model:
%   - 'N' for normal distribution;
%   - 'T' for Student's T distribution.
%
% [OUTPUT]
% g = A boolean n^2-by-n matrix representing the posterior density orthants.
% p = A column vector of floats [0,1] of length n^2 representing the posterior density probabilities.

function [g,p] = cimdo(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('r',@(x)validateattributes(x,{'double'},{'real' 'finite' '2d' 'nonempty'}));
        ip.addRequired('pods',@(x)validateattributes(x,{'double'},{'real' 'finite' 'vector' 'nonempty'}));
        ip.addRequired('md',@(x)any(validatestring(x,{'N' 'T'})));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    [r,pods] = validate_input(ipr.r,ipr.pods);
    md = ipr.md;

    nargoutchk(2,2);

    [g,p] = cimdo_internal(r,pods,md);

end

function [g,p] = cimdo_internal(r,pods,md)

    persistent options_mvncdf;
    persistent options_mvtcdf;
    persistent options_objective;

    if (isempty(options_mvncdf))
        options_mvncdf = optimset(optimset(@fsolve),'Algorithm','trust-region-dogleg','Diagnostics','off','Display','off','Jacobian','on');
    end

    if (isempty(options_mvtcdf))
        options_mvtcdf = optimset(optimset(@fsolve),'Algorithm','trust-region-dogleg','Diagnostics','off','Display','off','Jacobian','off');
    end

    if (isempty(options_objective))
        options_objective = optimset(optimset(@fsolve),'Display','none','TolFun',1e-6,'TolX',1e-6);
    end

    up = isempty(getCurrentTask());

    [t,n] = size(r);
    k = 2^n;
    g = dec2bin(0:(2^n - 1),n) - '0';

    rn = (r - repmat(mean(r),t,1)) ./ repmat(std(r),t,1);
    c = cov(rn);

    q = NaN(k,1);

    if (strcmp(md,'N'))
        dts = norminv(1 - pods);

        if (up)
            parfor i = 1:k
                g_i = g(i,:).';
                lb = min([(-Inf * ~g_i) dts],[],2);
                ub = max([(Inf * g_i) dts],[],2);

                q(i) = mvncdf_fast(c,lb,ub,options_mvncdf);
            end
        else
            for i = 1:k
                g_i = g(i,:).';
                lb = min([(-Inf * ~g_i) dts],[],2);
                ub = max([(Inf * g_i) dts],[],2);

                q(i) = mvncdf_fast(c,lb,ub,options_mvncdf);
            end
        end
    else
        params = mle(rn(:),'Distribution','tLocationScale');
        df = max(1,min(params(3),6));

        dts = tinv(1 - pods,df);

        if (up)
            parfor i = 1:k
                g_i = g(i,:).';
                lb = min([(-Inf * ~g_i) dts],[],2);
                ub = max([(Inf * g_i) dts],[],2);

                q(i) = mvtcdf_fast(c,df,lb,ub,options_mvtcdf);
            end
        else
            for i = 1:k
                g_i = g(i,:).';
                lb = min([(-Inf * ~g_i) dts],[],2);
                ub = max([(Inf * g_i) dts],[],2);

                q(i) = mvtcdf_fast(c,df,lb,ub,options_mvtcdf);
            end
        end
    end

    if (any(isnan(q)))
        p = NaN(k,1);
        return;
    end

    q = q ./ sum(q);

    x0 = zeros(n + 1,1);
    [sol,~,ef] = fsolve(@(x)objective(x,n,pods,g,q),x0,options_objective);

    if (ef ~= 1)
        p = NaN(k,1);
        return;
    end

    [~,p] = objective(sol,n,pods,g,q);
    p = p ./ sum(p);

end

function [cp,l,u] = cholperm(n,c,l,u)

    s2p = sqrt(2 * pi());

    cp = zeros(n,n);
    z = zeros(n,1);

    for i = 1:n
        i_seq = 1:(i - 1);
        in_seq = i:n;
        ipn_seq = (i + 1):n;

        cp_off = cp(in_seq,i_seq);
        z_off = z(i_seq);
        cpz = cp_off * z_off;

        d = diag(c);
        s = d(in_seq) - sum(cp_off.^2,2);
        s(s < 0) = eps();
        s = sqrt(s);

        lt = (l(in_seq) - cpz) ./ s;
        ut = (u(in_seq) - cpz) ./ s;

        p = Inf(n,1);
        p(in_seq) = logprobs(lt,ut);

        [~,k] = min(p);
        jk = [i k];
        kj = [k i];

        c(jk,:) = c(kj,:);
        c(:,jk) = c(:,kj);

        cp(jk,:) = cp(kj,:);
        l(jk) = l(kj);
        u(jk) = u(kj);

        s = c(i,i) - sum(cp(i,i_seq).^2);
        s(s < 0) = eps();

        cp(i,i) = sqrt(s);
        cp(ipn_seq,i) = (c(ipn_seq,i) - (cp(ipn_seq,i_seq) * (cp(i,i_seq)).')) / cp(i,i);

        cp_ii = cp(i,i);
        cpz = cp(i,i_seq) * z(i_seq);
        lt = (l(i) - cpz) / cp_ii;
        ut = (u(i) - cpz) / cp_ii;

        w = logprobs(lt,ut);
        z(i) = (exp((-0.5 * lt.^2) - w) - exp((-0.5 * ut.^2) - w)) / s2p;
    end

end

function p = logprobs(a,b)

    p = zeros(size(a));
    l2 = log(2);
    s2 = sqrt(2);

    a_indices = a > 0;

    if (any(a_indices))
        x = a(a_indices);
        pa = (-0.5 * x.^2) - l2 + reallog(erfcx(x / s2));

        x = b(a_indices);
        pb = (-0.5 * x.^2) - l2 + reallog(erfcx(x / s2));

        p(a_indices) = pa + log1p(-exp(pb - pa));
    end

    b_indices = b < 0;

    if (any(b_indices))
        x = -a(b_indices);
        pa = (-0.5 * x.^2) - l2 + reallog(erfcx(x / s2));

        x = -b(b_indices);
        pb = (-0.5 * x.^2) - l2 + reallog(erfcx(x / s2));

        p(b_indices) = pb + log1p(-exp(pa - pb));
    end

    indices = ~a_indices & ~b_indices;

    if (any(indices))
        pa = erfc(-a(indices) / s2) / 2;
        pb = erfc(b(indices) / s2) / 2;
        p(indices) = log1p(-pa - pb);
    end

end

function y = mvncdf_fast(c,lb,ub,options)

    n = size(c,1);

    [cp,lb,ub] = cholperm(n,c,lb,ub);
    d = diag(cp);

    if any(d < eps())
        y = NaN;
        return;
    end

    lb = lb ./ d;
    ub = ub ./ d;
    cp = (cp ./ repmat(d,1,n)) - eye(n);

    [sol,~,ef] = fsolve(@(x)mvncdf_fast_psi(x,cp,lb,ub),zeros(2 * (n - 1),1),options);

    if (ef ~= 1)
        y = NaN;
        return;
    end

    x = sol(1:(n - 1));
    x(n) = 0;

    mu = sol(n:((2 * n) - 2));
    mu(n) = 0;

    c = cp * x;
    lb = lb - mu - c;
    ub = ub - mu - c;

    psi = sum(logprobs(lb,ub) + (0.5 * mu.^2) - (x .* mu));

    y = exp(psi);

end

function [g,j] = mvncdf_fast_psi(y,cp,lb,ub)

    d = size(cp,1);
    d_seq = 1:(d - 1);

    x = zeros(d,1);
    x(d_seq) = y(d_seq);

    mu = zeros(d,1);
    mu(d_seq) = y(d:end);

    c = zeros(d,1);
    c(2:d) = cp(2:d,:) * x;

    lt = lb - mu - c;
    ut = ub - mu - c;

    w = logprobs(lt,ut);
    pd = sqrt(2 * pi());
    pl = exp((-0.5 * lt.^2) - w) / pd;
    pu = exp((-0.5 * ut.^2) - w) / pd;
    p = pl - pu;

    dfdx = -mu(d_seq) + (p.' * cp(:,d_seq)).';
    dfdm = mu - x + p;
    g = [dfdx; dfdm(d_seq)];

    lt(isinf(lt)) = 0;
    ut(isinf(ut)) = 0;

    dp = -p.^2 + (lt .* pl) - (ut .* pu);
    dl = repmat(dp,1,d) .* cp;

    mx = -eye(d) + dl;
    mx = mx(d_seq,d_seq);

    xx = cp.' * dl;
    xx = xx(d_seq,d_seq);

    j = [xx mx.'; mx diag(1 + dp(d_seq))];

end

function y = mvtcdf_fast(c,df,lb,ub,options)

    n = size(c,1);

    [cp,lb,ub] = cholperm(n,c,lb,ub);
    d = diag(cp);

    if any(d < eps())
        y = NaN;
        return;
    end

    lb = lb ./ d;
    ub = ub ./ d;
    cp = (cp ./ repmat(d,1,n)) - eye(n);

    x0 = zeros(2 * n,1);
    x0(2 * n) = sqrt(df);
    x0(n) = log(sqrt(df));

    [sol,~,ef] = fsolve(@(x)mvtcdf_fast_psi(x,cp,df,lb,ub),x0,options);

    if (ef ~= 1)
        y = NaN;
        return;
    end

    sol(n) = exp(sol(n));

    x = sol(1:n);
    r = x(n);
    x(n) = 0;

    mu = sol((n + 1):end); 
    eta = mu(n);
    mu(n) = 0;

    c = cp * x;
    lb = (r * (lb ./ sqrt(df))) - mu - c;
    ub = (r * (ub ./ sqrt(df))) - mu - c;

    psi = sum(logprobs(lb,ub) + (0.5 .* mu.^2) - (x .* mu)) + (log(2 * pi()) / 2) - gammaln(df / 2) - (log(2) * ((0.5 * df) - 1));
    psi = min(psi + (0.5 * eta^2) - (r * eta) + ((df - 1) * reallog(r)) + logprobs(-Inf,eta),0);

    y = exp(psi);

end

function g = mvtcdf_fast_psi(y,cp,df,lb,ub)

    d = size(cp,1);
    d_seq = 1:(d - 1);

    x = zeros(d,1);
    x(d_seq) = y(d_seq);

    mu = zeros(d,1);
    mu(d_seq) = y(d+1:2*d-1);

    r = exp(y(d));
    eta = y(2 * d);

    lb = lb ./ sqrt(df);
    ub = ub ./ sqrt(df);

    c = zeros(d,1);
    c(2:d) = cp(2:d,:) * x;

    lt = (r * lb) - mu - c;
    ut = (r * ub) - mu - c;

    w = logprobs(lt,ut);
    pd = sqrt(2 * pi());
    pl = exp((-0.5 * lt.^2) - w) / pd;
    pu = exp((-0.5 * ut.^2) - w) / pd;
    p = pl - pu;

    dfdx = -mu(d_seq) + (p.' * cp(:,d_seq)).';
    dfdm = mu - x + p;

    lb(isinf(lb)) = 0;
    ub(isinf(ub)) = 0;

    dfdr = ((df - 1) / r) - eta + sum((ub .* pu) - (lb .* pl));
    dfde = eta - r + exp((-0.5 * eta^2) - logprobs(-Inf,eta)) / pd;

    g = [dfdx; dfdm(d_seq); dfdr; dfde];

end

function [f,p] = objective(x,n,pods,g,q)

    mu = x(1);
    lambda = x(2:end);

    p = zeros(n + 1,1);

    for i = 1:numel(q)
        p(i) = q(i) * exp(-(1 + mu + (g(i,:) * lambda)));
    end

    lhs = [sum(p); sum(g .* repmat(p,1,n)).'];
    rhs = [1; pods];

    f = lhs - rhs;

end

function [r,pods] = validate_input(r,pods)

    [t,n] = size(r);

    if ((t < 5) || (n < 2))
        error('The value of ''r'' is invalid. Expected input to be a matrix with a minimum size of 5x2.');
    end

    pods = pods(:);

    if (numel(pods) ~= n)
        error(['The value of ''pods'' is invalid. Expected input to be a vector containing at least ' num2str(n) ' elements.']);
    end

end
