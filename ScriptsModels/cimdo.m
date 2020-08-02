% [INPUT]
% r = A float t-by-n matrix representing the logarithmic returns.
% pods = A vector of floats of length n representing the probabilities of default.
%
% [OUTPUT]
% g = An n^2-by-n matrix of numeric booleans representing the multivariate posterior density orthants.
% p = A vector of floats of length n^2 representing the multivariate posterior density probabilities.

function [g,p] = cimdo(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('r',@(x)validateattributes(x,{'double'},{'real' 'finite' '2d' 'nonempty'}));
        ip.addRequired('pods',@(x)validateattributes(x,{'double'},{'real' 'finite' 'vector' 'nonempty'}));
    end

    ip.parse(varargin{:});
    
    ipr = ip.Results;
    [r,pods] = validate_input(ipr.r,ipr.pods);

    nargoutchk(2,2);

    [g,p] = cimdo_internal(r,pods);

end

function [g,p] = cimdo_internal(r,pods)

    persistent options_cimdo;
    persistent options_mvncdf;

    if (isempty(options_cimdo))
        options_cimdo = optimset(optimset(@fsolve),'Display','none','TolFun',1e-6,'TolX',1e-6);
    end

    if (isempty(options_mvncdf))
        options_mvncdf = optimset(optimset(@fsolve),'Algorithm','trust-region-dogleg','Diagnostics','off','Display','off','Jacobian','on');
    end

    [t,n] = size(r);
    k = 2^n;
    
    g = dec2bin(0:(2^n - 1),n) - '0';
    
    rn = (r - repmat(mean(r),t,1)) ./ repmat(std(r),t,1);

    c = cov(rn);
    dts = norminv(1 - pods);

    q = NaN(k,1);

    for i = 1:k
        g_i = g(i,:).';
        lb = min([(-Inf * ~g_i) dts],[],2);
        ub = max([(Inf * g_i) dts],[],2);

        q(i) = mvncdf_fast(c,lb,ub,options_mvncdf);
    end
    
    if (any(isnan(q)))
        p = NaN(k,1);
        return;
    end
    
    q = q ./ sum(q);

    try
        f = fsolve(@(x)objective(x,n,pods,g,q),zeros(n + 1,1),options_cimdo);
        [~,p] = objective(f,n,pods,g,q);
        p = p ./ sum(p);
    catch
        p = NaN(k,1);
    end

end

function y = mvncdf_fast(c,lb,ub,options)

    n = size(c,1);

    [cp,lb,ub] = mvncdf_fast_cholperm(n,c,lb,ub);
    d = diag(cp);

    if any(d < eps())
        y = NaN;
        return;
    end

    lb = lb ./ d;
    ub = ub ./ d;
    cp = (cp ./ repmat(d,1,n)) - eye(n);

    [sol,~,exitflag] = fsolve(@(x)mvncdf_fast_psi(x,cp,lb,ub),zeros(2 * (n - 1),1),options);

    if (exitflag ~= 1)
        y = NaN;
        return;
    end

	x = sol(1:(n - 1));
    x(n) = 0;
    x = x(:);
    
    mu = sol(n:((2 * n) - 2));
    mu(n) = 0;
    mu = mu(:);
    
    c = cp * x;
    lb = lb - mu - c;
    ub = ub - mu - c;

    y = exp(sum(mvncdf_fast_logprobs(lb,ub) + (0.5 * mu.^2) - (x .* mu)));

end

function [cp,l,u] = mvncdf_fast_cholperm(n,c,l,u)

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
        p(in_seq) = mvncdf_fast_logprobs(lt,ut);

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

        w = mvncdf_fast_logprobs(lt,ut);
        z(i) = (exp((-0.5 * lt.^2) - w) - exp((-0.5 * ut.^2) - w)) / s2p;
    end

end

function p = mvncdf_fast_logprobs(a,b)

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

function [g,j] = mvncdf_fast_psi(y,cp,lb,ub)

    d = length(ub);
    d_seq = 1:(d - 1);

    x = zeros(d,1);
    x(d_seq) = y(d_seq);

    mu = zeros(d,1);
    mu(d_seq) = y(d:end);

    c = zeros(d,1);
    c(2:d) = cp(2:d,:) * x;

    lt = lb - mu - c;
    ut = ub - mu - c;

    w = mvncdf_fast_logprobs(lt,ut);
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
        error('The value of ''r'' is invalid. Expected input to have a minimum size of size 5x2.');
    end
    
    pods = pods(:);
    
    if (numel(pods) ~= n)
        error(['The value of ''pods'' is invalid. Expected input to be an array of ' num2str(n) ' elements.']);
    end

end
