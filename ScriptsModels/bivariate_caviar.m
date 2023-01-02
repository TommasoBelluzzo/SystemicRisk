% [INPUT]
% r = A float t-by-2 matrix (-Inf,Inf) representing the logarithmic returns, in which:
%   - the first column represents the market returns;
%   - the second column represents the firm returns.
% a = A float [0.01,0.10] representing the target quantile.
%
% [OUTPUT]
% caviar = A column vector of floats [0,Inf) of length t representing the Conditional Autoregressive Value at Risk.
% beta = A column vector of floats (-Inf,Inf) of length 10 representing the model coefficients.
% ir_fm = A float t-by-3 matrix (-Inf,Inf) representing the reaction of the firm to a shock of the market, in which:
%   - the first column represents the impulse response;
%   - the second column represents the lower bound;
%   - the third column represents the upper bound.
% ir_mf = A float t-by-3 matrix (-Inf,Inf) representing the reaction of the market against to shock of the firm, in which:
%   - the first column represents the impulse response;
%   - the second column represents the lower bound;
%   - the third column represents the upper bound.
% se = A column vector of floats [0,Inf) of length 10 representing model standard errors.
% stats = A row vector of floats [0,Inf) of length 2 representing model error statistics, where the first element is the critical value and the second element is the p-value.

function [caviar,beta,ir_fm,ir_mf,se,stats] = bivariate_caviar(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('r',@(x)validateattributes(x,{'double'},{'real' 'finite' '2d' 'nonempty' 'size' [NaN 2]}));
        ip.addRequired('a',@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.01 '<=' 0.10 'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    r = validate_input(ipr.r);
    a = ipr.a;

    nargoutchk(2,6);

    switch (nargout)
        case 3
            error('Both impulse response outputs must be assigned.');
        case 5
            error('Both standard error outputs must be assigned.');
    end

    cir = false;
    cse = false;

    if (nargout >= 3)
        cir = true;

        if (nargout >= 5)
            cse = true;
        end
    end

    [caviar,beta,ir_fm,ir_mf,se,stats] = bivariate_caviar_internal(r,a,cir,cse);

end

function [caviar,beta,ir_fm,ir_mf,se,stats] = bivariate_caviar_internal(r,a,cir,cse)

    persistent options;

    if (isempty(options))
        options = optimset(optimset(@fminsearch),'Display','none','MaxFunEvals',1000,'MaxIter',1000,'TolFun',1e-8,'TolX',1e-8);
    end

    up = isempty(getCurrentTask());

    rng_current = rng();
    rng(double(bitxor(bitxor(uint16('T'),uint16('B')),bitxor(uint16('B'),uint16('C')))));
    um_beta0 = unifrnd(0,1,[10000 3]);
    rng(rng_current);

    c = zeros(3,2);
    q = zeros(1,2);

    if (up)
        if (size(r,1) >= 200)
            qo = round(100 * a,0);

            parfor i = 1:2
                r_i = r(:,i);
                rhs_i = sort(r_i(1:100));
                q_i = rhs_i(qo);

                c(:,i) = univariate_model(r_i,q_i,a,um_beta0,options);
                q(i) = q_i;
            end
        else
            parfor i = 1:2
                r_i = r(:,i);
                q_i = quantile(r_i,a);

                c(:,i) = univariate_model(r_i,q_i,a,um_beta0,options);
                q(i) = q_i;
            end
        end
    else
        if (size(r,1) >= 200)
            qo = round(100 * a,0);

            for i = 1:2
                r_i = r(:,i);
                rhs_i = sort(r_i(1:100));
                q_i = rhs_i(qo);

                c(:,i) = univariate_model(r_i,q_i,a,um_beta0);
                q(i) = q_i;
            end
        else
            for i = 1:2
                r_i = r(:,i);
                q_i = quantile(r_i,a);

                c(:,i) = univariate_model(r_i,q_i,a,um_beta0);
                q(i) = q_i;
            end
        end
    end

    k1 = diag(c(2,:));
    k2 = diag(c(3,:));

    beta0 = [c(1,:).'; k1(:); k2(:)];
    beta1 = fminsearch(@(x)objective(x,r,a,q),beta0,options);
    beta = fminsearch(@(x)objective(x,r,a,q),beta1,options);

    [~,~,caviar_full] = objective(beta,r,a,q);
    caviar = -1 .* min(caviar_full(:,2),0);

    if (cir)
        [vc,se,stats] = standard_errors(r,a,beta,caviar_full);
        ir_fm = impulse_response(r,beta,vc,true);
        ir_mf = impulse_response(r,beta,vc,false);

        if (~cse)
            se = [];
            stats = [];
        end
    else
        ir_fm = [];
        ir_mf = [];
        se = [];
        stats = [];
    end

end

function ir = impulse_response(r,beta,vc,mkt)

    if (mkt)
        shock = [2; 0];
    else
        shock = [0; 2];
    end

    c = cov(r);
    cl = chol(c,'lower');

    b = reshape(beta,2,5);
    m1 = b(:,2:3);
    m2 = b(:,4:end);

    ir_all = zeros(200,2);
    ir_all(1,:) = m1 * abs(cl * shock);

    for i = 2:200
        ir_all(i,:) = m2 * ir_all(i-1,:).';
    end

    e2 = eye(2);
    e4 = eye(4);
    z42 = zeros(4,2);
    z44 = zeros(4);

    da = [z42 e4 z44];
    db = [z42 z44 e4];

    g1 = kron(abs(cl * shock).',e2) * da;
    g2 = (m2 * g1) + kron((m1 * abs(cl * shock)).',e2) * db;

    se = zeros(200,2);
    se(1,:) = diag(sqrt(g1 * vc * g1.'));
    se(2,:) = diag(sqrt(g2 * vc * g2.'));

    for i = 3:200
        sb = zeros(4);

        for j = 0:i-2
            sb = sb + kron((m2.')^(i-2-j),m2^j);
        end

        g = (m2^(i-1) * g1) + kron((m1 * abs(cl * shock)).',e2) * (sb * db);
        se(i,:) = diag(sqrt(g * vc * g.')); 
    end

    ir_all_lb = ir_all - (2 .* se);
    ir_all_ub  = ir_all + (2 .* se);

    if (mkt)
        ir = [ir_all(:,2) ir_all_lb(:,2) ir_all_ub(:,2)];
    else
        ir = [ir_all(:,1) ir_all_lb(:,1) ir_all_ub(:,1)];
    end

end

function [rq,hits,caviar] = objective(beta,r,a,q)

    t = size(r,1);

    caviar = zeros(t,2);
    caviar(1,:) = q;

    b = reshape(beta,2,5);
    m0 = b(:,1).';
    m1 = b(:,2:3).';
    m2 = b(:,4:end).';

    for t = 2:t
        caviar(t,:) = m0 + (abs(r(t-1,:)) * m1) + (caviar(t-1,:) * m2);
    end

    hits = (a * ones(t,2)) - (r < caviar);
    rq = mean(sum((r - caviar) .* hits,2));

    if (~isfinite(rq))
        rq = 1e100;
    end

end

function [vc,se,stats] = standard_errors(r,a,beta,caviar)

    t = size(r,1);

    e2 = eye(2);

    m = reshape(beta,2,5);
    dm1 = [e2 zeros(2,8)];
    dm2 = [zeros(4,2) eye(4) zeros(4)];
    dm3 = [zeros(4,6) eye(4)];

    dq = zeros(2,10,t);

    for t = 2:t
        dq(:,:,t) = dm1 + (kron(abs(r(t-1,:)),e2) * dm2) + (m(:,4:end) * dq(:,:,t-1)) + (kron(caviar(t-1,:),e2) * dm3);
    end

    d = r - caviar;

    k = median(abs(d(:,1) - median(d(:,1))));
    h = t^(-1/3) * norminv(0.975)^(2/3) * ((1.5 * normpdf(norminv(a))^2) / ((2 * norminv(a)^2) + 1))^(1/3);
    c = k * (norminv(a + h) - norminv(a - h));

    q = zeros(10);
    v = zeros(10);

    for t = 1:t
        psi = a - (d(t,:) < 0).';
        eta = sum(reshape(dq(:,:,t),2,10) .* (psi * ones(1,10)));

        v = v + (eta.' * eta);

        qt = zeros(10);

        for j = 1:2
            dqt = reshape(dq(j,:,t),10,1);
            qt = qt + ((abs(d(t,j)) < c) * (dqt * dqt.'));
        end

        q = q + qt;
    end

    q = q / (2 * c * t);
    v = v / t; 
    vc = (linsolve(q,v) / q) / t;

    r = [zeros(4,3), [e2; zeros(2)], zeros(4,2), [zeros(2); e2], zeros(4,1)];
    cv = ((r * beta).' / (r * vc * r.')) * (r * beta);
    pval = 1 - chi2cdf(cv,4);

    se = sqrt(diag(vc));
    stats = [cv pval];

end

function beta = univariate_model(r,q,a,beta0,options)

    w = size(beta0,1);
    rq0 = zeros(w,1);

    for i = 1:w
        [rq0(i),~,~] = univariate_model_objective(beta0(i,:).',r,a,q);
    end

    m = [rq0 beta0];
    ms = sortrows(m,1);

    beta1  = ms(1,2:end).';
    beta2 = fminsearch(@(x)univariate_model_objective(x,r,a,q),beta1,options);
    beta = fminsearch(@(x)univariate_model_objective(x,r,a,q),beta2,options);

end

function [rq,hits,caviar] = univariate_model_objective(beta,r,a,q)

    t = numel(r);

    caviar = zeros(t,1);
    caviar(1) = q;

    for t = 2:t
        caviar(t) = beta(1) + (beta(2) * abs(r(t-1))) + (beta(3) * caviar(t-1));
    end

    hits = -((r < caviar) - a);
    rq  = hits.' * (r - caviar);

    if (~isfinite(rq))
        rq = 1e100;
    end

end

function r = validate_input(r)

    t = size(r,1);

    if (t < 5)
        error('The value of ''r'' is invalid. Expected input to be a matrix with at least 5 rows.');
    end

end
