% [INPUT]
% p = A vector of floats [0,Inf) of length t representing the prices.
% bw = An integer [21,252] representing the dimension of each rolling window.
% w = An integer [500,Inf) representing the number of sweeps (optional, default=1000).
% c = A float (0,Inf) representing the starting coefficient value (optional, default=0.01).
% s2 = A float (0,Inf) representing the starting variance of innovations (optional, default=0.0004).
%
% [OUTPUT]
% ris = A column vector of floats (0,Inf) of length t representing the Roll implicit spread.

function ris = roll_implicit_spread(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('p',@(x)validateattributes(x,{'double'},{'real' 'finite' 'nonnegative' 'vector' 'nonempty'}));
        ip.addRequired('bw',@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 21 '<=' 252 'scalar'}));
        ip.addOptional('w',1000,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 500}));
        ip.addOptional('c',0.01,@(x)validateattributes(x,{'double'},{'real' 'finite' 'positive'}));
        ip.addOptional('s2',0.0004,@(x)validateattributes(x,{'double'},{'real' 'finite' 'positive'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    p = validate_input(ipr.p);
    bw = ipr.bw;
    w = ipr.w;
    c = ipr.c;
    s2 = ipr.s2;

    nargoutchk(1,1);

    ris = roll_implicit_spread_internal(p,bw,w,c,s2);

end

function ris = roll_implicit_spread_internal(p,bw,w,c,s2)

    up = isempty(getCurrentTask());

    windows = extract_rolling_windows(log(max(1e-6,p)),bw);
    ris = zeros(numel(windows),1);

    if (up)
        parfor i = 1:numel(windows)
            ris(i) = gibbs_sampler(windows{i},w,c,s2);
        end
    else
        for i = 1:numel(windows)
            ris(i) = gibbs_sampler(windows{i},w,c,s2);
        end
    end

    alpha = 2 / (bw + 1);

    ris = [ris(1); filter(alpha,[1 (alpha - 1)],ris(2:end),(1 - alpha) * ris(1))];

end

function g = gibbs_sampler(p,w,c,s2)

    dp = diff(p);
    q = [1; sign(dp)];

    for i = 1:w
        dq = diff(q);

        d = 1 + ((1 / s2) * (dq.' * dq));
        mu = linsolve(d,(1 / s2) * (dq.' * dp));
        rho = inv(d);
        c = mvnrnd_truncated(mu,rho);

        u = dp - (c .* dq);
        alpha = 1e-12 + (numel(u) / 2);
        beta = 1e-12 + (sum(u.^2) / 2);
        s2 = 1 / ((1 / beta) * gamrnd(alpha,1));

        q = perform_draw(p,q,c,s2);
    end

    g = 2 * c;

end

function q = perform_draw(p,q,c,s2)

    t = numel(p);
    q_nnz = q ~= 0;
    s22 = s2 * 2;

    m = mod((1:t).',2);
    r = rand(t,1);

    q = [q q];
    p = [p p];

    for i = 1:2
        o = (m == (i - 1)) & q_nnz;

        if (~any(o))
            continue;
        end

        q(o,1) = 1;
        q(o,2) = -1;

        u = diff(p - (c * q));

        s = u.^2 ./ s22;
        s_sum = [s; [0 0]] + [[0 0]; s];
        s_sum = s_sum(o,:);

        odds_log = diff(s_sum,1,2);
        in_range = odds_log < 500;
        odds = exp(in_range .* odds_log);

        buy = odds ./ (1 + odds);
        buy = (in_range .* buy) + ~in_range;

        if (i == 1)
            q(o,:) = repmat(1 - (2 * (r(o) > buy)),1,2);
        else
            q(o,1) = 1 - (2 * (r(o) > buy));
        end
    end

    q = q(:,1);

end

function r = mvnrnd_truncated(mu,rho)

    f = sqrt(rho);
    low = -mu / f;

    if (low > 6)
        eta = low + (100 * eps());
    else
        plow = normcdf(low);

        p = plow + (rand() * (1 - plow));

        if (p == 1)
            eta = low + (100 * eps());
        else
            eta = norminv(p);
        end
    end

    r = mu + (f * eta);

end

function p = validate_input(p)

    p = p(:);

    if (numel(p) < 5)
        error('The value of ''p'' is invalid. Expected input to be a vector containing at least 5 elements.');
    end

end
