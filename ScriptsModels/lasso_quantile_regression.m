% [INPUT]
% y = A vector of floats (-Inf,Inf) of length t representing the dependent variable.
% x = A float t-by-n matrix (-Inf,Inf) representing the independent variables.
% a = A float [0.01,0.10] representing the target quantile.
% ms = An integer [50,1000] representing the maximum number of steps (optional, default=100).
%
% [OUTPUT]
% beta = A row vector of floats (-Inf,Inf) representing the Beta coefficients.
% lambda = A float (-Inf,0] representing the L1-norm penalization term.

function [beta,lambda] = lasso_quantile_regression(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('y',@(x)validateattributes(x,{'double'},{'real' 'finite' 'vector' 'nonempty'}));
        ip.addRequired('x',@(x)validateattributes(x,{'double'},{'real' 'finite' '2d' 'nonempty'}));
        ip.addRequired('a',@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.01 '<=' 0.10 'scalar'}));
        ip.addOptional('ms',100,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 50 '<=' 1000 'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    [y,x] =  validate_input(ipr.y,ipr.x);
    a = ipr.a;
    ms = ipr.ms;

    nargoutchk(2,2);

    [beta,lambda] = lasso_quantile_regression_internal(y,x,a,ms);

end

function [beta,lambda] = lasso_quantile_regression_internal(y,x,a,ms)

    [t,n] = size(x);

    if (numel(unique(y)) < t)
        yd = abs(diff(sort(y)));
        yd(yd == 0) = [];

        yv = max(min(yd) / 100,1e-10);
        y = y + normrnd(0,yv,[t 1]);
    end

    idx_t = 1:t;
    idx_n = 1:n;

    [q,r,nu0,nu,lambda0,idx_v,idx_e,idx_l,idx_r] = initialize(y,x,a);

    fit = zeros(ms+1,t);

    beta = zeros(ms+1,n);
    beta0 = [q; zeros(ms,1)];

    o = ones(1,t) .* q;
    fit(1,:) = o;
    gacv = [(gacv_fit(o,y,a) / t); zeros(ms,1)];

    lambda = [lambda0; zeros(ms,1)];

    k = 0;
    drop = false;

    while (k < ms)
        k = k + 1;

        exc_t = idx_t(~ismember(idx_t,idx_e));
        gamma = nu0 + (x(exc_t,idx_v) * nu);

        delta1 = r(exc_t) ./ gamma;

        if (all(delta1 <= 0))
            delta = Inf;
        else
            delta = min(delta1(delta1 > 1e-10),[],'omitnan');
        end

        if (k > 1)
            delta2 = -beta(k,idx_v) ./ nu.';

            if (all(delta2 <= 0))
                z = Inf;
            else
                z = min(delta2(delta2 > 1e-10),[],'omitnan');
            end

            if (z < delta)
                drop = true;
                delta = z;
            else
                drop = false;
            end
        end

        if (isinf(delta))
            break;
        end

        if (drop)
            idx_tmp = idx_v(delta2 > 1e-10);
            delta_tmp = delta2(delta2 > 1e-10);
            [~,idx] = min(delta_tmp);
            j_drop = find(idx_v == idx_tmp(idx));
        else
            idx_tmp = exc_t(delta1 > 1e-10);
            delta_tmp = delta1(delta1 > 1e-10);
            [~,idx] = min(delta_tmp);
            i_star = idx_tmp(idx);
        end

        r(exc_t) = r(exc_t) - (delta * gamma);

        beta(k+1,:) = beta(k,:);
        beta(k+1,idx_v) = beta(k+1,idx_v) + (delta .* nu).';
        beta0(k+1) = beta0(k) + (delta * nu0);

        o = beta0(k+1) + (beta(k+1,:) * x.');
        fit(k+1,:) = o;
        gacv(k+1) = gacv_fit(o,y,a) / (t - numel(idx_v));

        if (~drop && (sum(idx_l) + sum(idx_r) == 1) || (numel(idx_v) == n))
            lambda(k) = 0;
            break;
        end

        exc_n = idx_n(~ismember(idx_n,idx_v));

        tmp_e = idx_e;
        tmp_l = idx_l;
        tmp_r = idx_r;

        if (drop)
            idx_v = idx_v(~ismember(idx_v,j_drop));
        else
            tmp_e = union(tmp_e,i_star,'stable');
            tmp_l = setdiff(idx_l,i_star);
            tmp_r = setdiff(idx_r,i_star);
        end

        ta = [[0 sign(beta(k+1,idx_v))]; [ones(numel(tmp_e),1) x(tmp_e,idx_v)]];
        tb = [1 zeros(1,numel(idx_v))];
        [var,nu_var,lambda_var] = calculate_vars(x,a,exc_n,ta,idx_v,tmp_e,tmp_l,tmp_r);

        obs = numel(tb);
        nu_obs = zeros(numel(idx_v) + 1,numel(tmp_e));
        lambda_obs = zeros(1,numel(tmp_e));
        left_obs = false(1,numel(tmp_e));

        for i = 1:numel(tmp_e)
            ta_i = ta;
            ta_i(i+1,:) = [];

            if (rank(ta_i) < obs)
                lambda_tmp = Inf;
            else
                nu_tmp = linsolve(ta_i,tb.');
                lambda_tmp = ((1 - a) * sum([ones(numel(tmp_l),1) x(tmp_l,idx_v)] * nu_tmp)) - (a * sum([ones(numel(tmp_r),1) x(tmp_r,idx_v)] * nu_tmp));

                nu_obs(:,i) = nu_tmp;

                yf = [1 x(tmp_e(i),idx_v)] * nu_tmp;

                if (yf > 0)
                    lambda_tmp = lambda_tmp + ((1 - a) * yf);
                    left_obs(i) = true;
                else
                    lambda_tmp = lambda_tmp - (a * yf);
                end
            end

            lambda_obs(i) = lambda_tmp;
        end

        lambda1 = min(lambda_var);
        lambda2 = min(lambda_obs);

        if ((lambda1 < lambda2) && (lambda1 < 0))
            [~,idx] = min(lambda_var);
            j_star = exc_n(idx);
            idx_v = union(idx_v,j_star,'stable');

            if (~drop)
                idx_e = union(idx_e,i_star,'stable');
                idx_l = setdiff(idx_l,i_star);
                idx_r = setdiff(idx_r,i_star);
            end

            nu0 = nu_var(1,idx);
            nu = nu_var(2:var,idx);
            lambda(k+1) = lambda1;
        elseif ((lambda2 < lambda1) && (lambda2 < 0))
            if (~drop)
                idx_l = setdiff(idx_l,i_star);
                idx_r = setdiff(idx_r,i_star);
            end

            [~,idx] = min(lambda_obs);
            i_star = tmp_e(idx);

            idx_e = tmp_e;
            idx_e(idx) = [];

            if (~left_obs(idx))
                idx_r = union(idx_r,i_star,'stable');
            else
                idx_l = union(idx_l,i_star,'stable');
            end

            nu0 = nu_obs(1,idx);
            nu = nu_obs(2:obs,idx);
            lambda(k+1) = lambda2;
        else
            break;
        end    

        if (abs(lambda(k+1)) < 1e-10)
            break;
        end

        drop = false;
    end

    gacv = gacv(1:k+1);
    [~,idx] = min(gacv);

	beta = beta(idx,:);
	lambda = min(lambda(idx),0);

end

function [var,nu_var,lambda_var] = calculate_vars(x,a,exc,tmp,idx_v,idx_e,idx_l,idx_r)

    m = numel(exc);

    if (isempty(tmp))
        f = @(tmp,j_star,s)[[0 1]; [s .* ones(numel(idx_e),1) x(idx_e,j_star)]];
    else
        f = @(tmp,j_star,s)[tmp [s * 1; x(idx_e,j_star)]];
    end

    b = [1; zeros(numel(idx_v) + 1,1)];

    var = numel(b);
	nu_var = zeros(var,m);
	lambda_var = Inf(m,1);

	for j = 1:m
        j_star = exc(j);

        z1 = f(tmp,j_star,1);

        if (rank(z1) < var)
            lambda1 = Inf;
        else
            nu1 = linsolve(z1,b);

            if (nu1(var) > 0)
                idx = [idx_v j_star];
                lambda1 = ((1 - a) * sum([ones(numel(idx_l),1) x(idx_l,idx)] * nu1)) - (a * sum([ones(numel(idx_r),1) x(idx_r,idx)] * nu1));
            else
                lambda1 = Inf;
            end
        end

        z2 = f(tmp,j_star,-1);

        if (rank(z2) < var)
            lambda2 = Inf;
        else
            nu2 = linsolve(z2,b);

            if (nu2(var) < 0)
                idx = [idx_v j_star];
                lambda2 = ((1 - a) * sum([ones(numel(idx_l),1) x(idx_l,idx)] * nu2)) - (a * sum([ones(numel(idx_r),1) x(idx_r,idx)] * nu2));
            else
                lambda2 = Inf;
            end
        end

        if (isinf(lambda1) && isinf(lambda2))
            lambda_var(j) = Inf;
        elseif (lambda1 > lambda2)
            nu_var(:,j) = nu2;
            lambda_var(j) = lambda2;
        else
            nu_var(:,j) = nu1;
            lambda_var(j) = lambda1;
        end
  end

end

function [q,r,nu0,nu,lambda0,idx_v,idx_e,idx_l,idx_r] = initialize(y,x,a)

    [t,n] = size(x);
    exc = 1:n;

    yc = 1:numel(y);
    ys = sort(y);

    q = ys(floor(t * a) + 1);
    r = y - q;

    idx_e = find(q == y,1,'last');
    idx_l = yc(y < y(idx_e));
    idx_r = yc(y > y(idx_e));

    [var,nu_var,lambda_var] = calculate_vars(x,a,exc,[],[],idx_e,idx_l,idx_r);

    [lambda0,idx] = min(lambda_var);
    idx_v = exc(idx);
    nu0 = nu_var(1,idx);
    nu = nu_var(2:var,idx);

end

function x = gacv_fit(beta0,y,a)

    z = y - beta0(:);
    x = sum(a * z(z > 0)) - sum((1 - a) * z(z <= 0));

end

function [y,x] = validate_input(y,x)

    y = y(:);
    ty = numel(y);

    if (ty < 5)
        error('The value of ''y'' is invalid. Expected input to be a vector containing at least 5 elements.');
    end

    if (isvector(x))
        x = x(:);
        tx = numel(x);

        if (tx ~= ty)
            error(['The value of ''x'' is invalid. Expected input to be a vector containing ' num2str(ty) ' elements.']);
        end
    else
        tx = size(x,1);

        if (tx ~= ty)
            error(['The value of ''x'' is invalid. Expected input to be a matrix containing ' num2str(ty) ' rows.']);
        end
    end

end
