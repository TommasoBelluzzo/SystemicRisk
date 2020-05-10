% [INPUT]
% p = A vector of floats (Inf,Inf) of length t representing the log prices.
% sw = An integer [500,Inf) representing the number of sweeps (optional, default=1000).
% c = A float (0,Inf) representing the starting coefficient value (optional, default=0.01).
% s2 = A float (0,Inf) representing the starting variance of innovations (optional, default=0.0004).
%
% [OUTPUT]
% is = A float (0,Inf) representing the implicit spread.

function is = roll_gibbs(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('p',@(x)validateattributes(x,{'double'},{'real','finite','vector','nonempty'}));
        ip.addOptional('sw',1000,@(x)validateattributes(x,{'double'},{'real','finite','integer','>=',500}));
        ip.addOptional('c',0.01,@(x)validateattributes(x,{'double'},{'real','finite','positive'}));
        ip.addOptional('s2',0.0004,@(x)validateattributes(x,{'double'},{'real','finite','positive'}));
    end

    ip.parse(varargin{:});
    
    ipr = ip.Results;
    p = ipr.p;
    sw = ipr.sw;
    c = ipr.c;
    s2 = ipr.s2;

    nargoutchk(1,1);

	is = roll_gibbs_internal(p,sw,c,s2);

end

function is = roll_gibbs_internal(p,sw,c,s2)

    p = p(:);
    dp = diff(p);

    q = [1; sign(dp)];

    for i = 1:sw
        dq = diff(q);

        d = 1 + ((1 / s2) * (dq.' * dq));
        mu = d \ ((1 / s2) * (dq.' * dp));
        rho = inv(d);
        c = truncated_mvnrnd(mu,rho);

        u = dp - (c .* dq);
        alpha = 1e-12 + (numel(u) / 2);
        beta = 1e-12 + (sum(u.^2) / 2);
        s2 = 1 / ((1 / beta) * gamrnd(alpha,1));

        q = perform_draw(p,q,c,s2);
    end
    
    is = 2 * c;

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

function r = truncated_mvnrnd(mu,rho)

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
