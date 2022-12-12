% [INPUT]
% dep = A vector of floats of length t representing the dependent variable.
% indep_s = A float t-by-cs matrix (-Inf,Inf) representing the switching independent variables, without intercept because it is internally handled by the model (optional, default=[]).
% indep_ns = A float t-by-cns matrix (-Inf,Inf) representing the non-switching independent variables (optional, default=[]).
% k = An integer [2,4] representing the number of states of the model (optional, default=2).
% vs = A boolean that indicates whether the variance is allowed to switch (optional, default=true).
% finit = A function handle representing a hook for generating a customized minimization problem (optional, by default a standard minimization problem is generated).
%   The function handle must accept the following input arguments (in the same order):
%     - x0: A column vector of floats of length cx representing the initial parameters.
%     - dep, indep_s, indep_s, k, vs, tmm: See the description above.
%     - p0 = A float k-by-k matrix [0,1] representing the initial stochastic row-wise transition matrix.
%     - options = A structure representing the optimization options structure.
%   The function handle must return the following output arguments (in the same order):
%     - x0 = A column vector of floats of length cx = (vs ? k : 1) + (cs * k) + cns representing the initial parameters for variance and independent variables (mandatory).
%     - ai = A float ci-by-cx matrix (-Inf,Inf) representing the "A" element of linear inequality constraints (optional, defaultable to []).
%     - bi = A column vector of floats of length ci representing the "b" element of linear inequality constraints (optional, defaultable to []).
%     - ae = A float ce-by-cx matrix (-Inf,Inf) representing the "Aeq" element of linear equality constraints (optional, defaultable to []).
%     - ce = A column vector of floats of length ce representing the "beq" element of linear equality constraints (optional, defaultable to []).
%     - lb = A column vector of floats of length cx representing the lower bounds (mandatory).
%     - ub = A column vector of floats of length cx representing the upper bounds (mandatory).
% tmm = A float k-by-k matrix [0,1] representing the mask of the stochastic row-wise transition matrix, in which NaNs represent the elements to be computed (optional, by default all the elements are computed).
% fnlcon = A function handle representing a hook for applying non-linear constraints (optional, by default no non-linear constraints are applied).
%   The function handle must accept the following input arguments (in the same order):
%     - x: A column vector of floats representing the current parameters.
%     - dep, indep_s, indep_s, k, vs, tmm: See the description above.
%     - options = A structure representing the optimization options structure.
%   The function handle must return the following output arguments (in the same order):
%     - ci = A column vector of floats representing the "C" element of non-linear constraints (optional, defaultable to []).
%     - ce = A column vector of floats representing the "Ceq" element of non-linear constraints (optional, defaultable to []).
%
% [OUTPUT]
% indep_s_params = A cs-by-1 cell array of row vectors of floats containing the parameters of switching independent variables.
% indep_ns_params = A row vector of floats of length cns representing the parameters of non-switching independent variables.
% s2_params = A row vector of floats of length k representing the variance parameters.
% p = A float k-by-k matrix [0,1] representing the stochastic row-wise transition matrix.
% sprob = A float t-by-k matrix [0,1] representing the smoothed probabilities of each state.
% dur = A row vector of floats of length 4 representing the duration of each state.
% cmu = A column vector of floats of length t representing the conditional means.
% cs2 = A column vector of floats of length t representing the conditional variances.
% e = A column vector of floats of length t representing the standardized residuals.

function [indep_s_params,indep_ns_params,s2_params,p,sprob,dur,cmu,cs2,e] = regime_switching(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('dep',@(x)validateattributes(x,{'double'},{'real' 'finite' 'column' 'nonempty'}));
        ip.addOptional('indep_s',[],@(x)validateattributes(x,{'double'},{'real' 'finite' '2d'}));
        ip.addOptional('indep_ns',[],@(x)validateattributes(x,{'double'},{'real' 'finite' '2d'}));
        ip.addOptional('k',2,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 2 '<=' 4 'scalar'}));
        ip.addOptional('vs',true,@(x)validateattributes(x,{'logical'},{'scalar'}));
        ip.addOptional('finit',[],@(x)validateattributes(x,{'double' 'function_handle'},{}));
        ip.addOptional('tmm',[],@(x)validateattributes(x,{'double'},{'real' '2d'}));
        ip.addOptional('fnlcon',[],@(x)validateattributes(x,{'double' 'function_handle'},{}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    vs = ipr.vs;
    [dep,indep_s,indep_ns,k,finit,tmm,p0,fnlcon] = validate_input(ipr.dep,ipr.indep_s,ipr.indep_ns,ipr.k,ipr.finit,ipr.tmm,ipr.fnlcon);

    nargoutchk(6,9);

    [indep_s_params,indep_ns_params,s2_params,p,sprob,dur,cmu,cs2,e] = regime_switching_internal(dep,indep_s,indep_ns,k,vs,finit,tmm,p0,fnlcon);

end

function [indep_s_params,indep_ns_params,s2_params,p,sprob,dur,cmu,cs2,e] = regime_switching_internal(dep,indep_s,indep_ns,k,vs,finit,tmm,p0,fnlcon)

    persistent options;

    if (isempty(options))
        options = optimset(optimset(@fmincon),'Algorithm','sqp','Diagnostics','off','Display','off','LargeScale','off','MaxSQPIter',1000,'TolFun',1e-6);
    end

    t = numel(dep);

    [p_params,p_lb,p_ub,p_ae,p_be] = parametrize_p(k,tmm,p0);
    p_params_count = numel(p_params);

    if (isempty(finit))
        x0 = [];
        [ai,bi] = deal([]);
        [ae,be] = deal([]);
        [lb,ub] = deal([]);

        [s2_params,s2_lb,s2_ub] = parametrize_s2(dep,k,vs,1.5,0.75);
        s2_params_count = numel(s2_params);
        x0 = [x0; s2_params];
        lb = [lb; s2_lb];
        ub = [ub; s2_ub];

        [indep_s_params,indep_s_lb,indep_s_ub] = parametrize_indep_s(dep,indep_s,k);
        indep_s_params_count = numel(indep_s_params);
        x0 = [x0; indep_s_params];
        lb = [lb; indep_s_lb];
        ub = [ub; indep_s_ub];

        [indep_ns_params,indep_ns_lb,indep_ns_ub] = parametrize_indep_ns(dep,indep_ns);
        indep_ns_params_count = numel(indep_ns_params);
        x0 = [x0; indep_ns_params];
        lb = [lb; indep_ns_lb];
        ub = [ub; indep_ns_ub];

        params_count = indep_s_params_count + indep_ns_params_count + s2_params_count;
    else
        [x0,ai,bi,ae,be,lb,ub] = finit(dep,indep_s,indep_ns,k,vs,tmm,p0,options);

        if (vs)
            s2_params_count = k;
        else
            s2_params_count = 1;
        end

        indep_s_params_count = size(indep_s,2) * k;
        indep_ns_params_count = size(indep_ns,2);

        params_count = s2_params_count + indep_s_params_count + indep_ns_params_count;

        if (~isa(x0,'double') || ~ismatrix(x0) || (size(x0,2) ~= 1) || (size(x0,1) ~= params_count))
            error(['The function ''finit'' generated an invalid value for ''x0'': it must be a float column vector of length ' num2str(params_count) '.']);
        end

        if (~isempty(ai) || ~isempty(bi))
            if (~isa(ai,'double') || ~ismatrix(ai))
                error(['The function ''finit'' generated an invalid value for ''ai'': it must be a float 2d matrix with no zero-valued dimensions and the number of columns equal to ' num2str(params_count) '.']);
            end

            [ai_r,ai_c] = size(ai);

            if ((ai_r == 0) || (ai_c ~= params_count))
                error(['The function ''finit'' generated an invalid value for ''ai'': it must be a float 2d matrix with no zero-valued dimensions and the number of columns equal to ' num2str(params_count) '.']);
            end

            if (~isa(bi,'double') || ~ismatrix(bi))
                error(['The function ''finit'' generated an invalid value for ''bi'': it must be a a column vector of length ' num2str(ai_r) '.']);
            end

            [bi_r,bi_c] = size(bi);

            if ((bi_r ~= ai_r) || (bi_c ~= 1))
                error(['The function ''finit'' generated an invalid value for ''bi'': it must be a a column vector of length ' num2str(ai_r) '.']);
            end

            ai = [ai zeros(ai_r,p_params_count)];
        end

        if (~isempty(ae) || ~isempty(be))
            if (~isa(ae,'double') || ~ismatrix(ae))
                error(['The function ''finit'' generated an invalid value for ''ae'': it must be a float 2d matrix with no zero-valued dimensions and the number of columns equal to ' num2str(params_count) '.']);
            end

            [ae_r,ae_c] = size(ae);

            if ((ae_r == 0) || (ae_c ~= params_count))
                error(['The function ''finit'' generated an invalid value for ''ae'': it must be a float 2d matrix with no zero-valued dimensions and the number of columns equal to ' num2str(params_count) '.']);
            end

            if (~isa(be,'double') || ~ismatrix(be))
                error(['The function ''finit'' generated an invalid value for ''be'': it must be a a column vector of length ' num2str(ae_r) '.']);
            end

            [be_r,be_c] = size(be);

            if ((be_r ~= ae_r) || (be_c ~= 1))
                error(['The function ''finit'' generated an invalid value for ''be'': it must be a a column vector of length ' num2str(ae_r) '.']);
            end

            ae = [ae zeros(ae_r,p_params_count)];
        end

        if (~isa(lb,'double') || ~ismatrix(lb) || (size(lb,2) ~= 1) || (size(lb,1) ~= params_count))
            error(['The function ''finit'' generated an invalid value for ''lb'': it must be a float column vector of length ' num2str(params_count) '.']);
        end

        if (~isa(ub,'double') || ~ismatrix(ub) || (size(ub,2) ~= 1) || (size(ub,1) ~= params_count))
            error(['The function ''finit'' generated an invalid value for ''ub'': it must be a float column vector of length ' num2str(params_count) '.']);
        end
    end

    x0 = [x0; p_params];
    ae = [ae; zeros(k,params_count) p_ae];
    be = [be; p_be];
    lb = [lb; p_lb];
    ub = [ub; p_ub];

    if (isempty(fnlcon))
        params = fmincon(@(x)likelihood(x,dep,indep_s,indep_ns,k,vs,tmm),x0,ai,bi,ae,be,lb,ub,[],options);
    else
        params = fmincon(@(x)likelihood(x,dep,indep_s,indep_ns,k,vs,tmm),x0,ai,bi,ae,be,lb,ub,@(x)fnlcon(x,dep,indep_s,indep_ns,k,vs,tmm,options),options);
    end

    [~,mu,g] = likelihood(params,dep,indep_s,indep_ns,k,vs,tmm);

    if (vs)
        s2_params = params(1:k).';
        o = k + 1;
    else
        s2_params = ones(1,k) .* params(1);
        o = 2;
    end

    indep_s_count = size(indep_s,2);
    indep_s_params = num2cell(reshape(params(o:o+indep_s_params_count-1),k,indep_s_count).',2);
    o = o + indep_s_params_count;

    if (indep_ns_params_count > 0)
        indep_ns_params = params(o:o+indep_ns_params_count-1).';
        o = o + indep_ns_params_count;
    else
        indep_ns_params = [];
    end

    p = tmm;
    p(isnan(tmm)) = params(o:end);

    pt = p.';

    prob = [ones(1,k) .* (1 / k); zeros(t - 1,k)];

    for i = 2:t
        prob(i,:) = pt * g(i-1,:).';
    end

    sprob = [zeros(t - 1,k); g(t,:)];

    for i = t-1:-1:1
        for j = 1:k
            sprob(i,j) = sum((sprob(i+1,:) .* g(i,j) .* pt(:,j).') ./ prob(i+1,:),'omitnan');
        end
    end

    dur = round(1 ./ (1 - diag(p).'),0);

    cmu = sum(mu .* prob,2);
    cs2 = sum(repmat(sqrt(s2_params),t,1) .* prob,2);

    e = dep - cmu;
    e = (e - mean(e)) ./ std(e);

end

function [ll,mu,g] = likelihood(x,dep,indep_s,indep_ns,k,vs,tmm)

    t = numel(dep);

    if (vs)
        c = x(1:k);
        o = k + 1;
    else
        c = ones(k,1) .* x(1);
        o = 2;
    end

    indep_s_count = size(indep_s,2);
    indep_s_params_count = indep_s_count * k;
    indep_s_params = reshape(x(o:o+indep_s_params_count-1),k,indep_s_count).';
    o = o + indep_s_params_count;

    indep_ns_count = size(indep_ns,2);

    if (indep_ns_count == 0)
        indep_ns = zeros(t,1);
        indep_ns_params = 0;
    else
        indep_ns_params = x(o:o+indep_ns_count-1);
        o = o + indep_ns_count;
    end

    p = tmm;
    p(isnan(tmm)) = x(o:end);

    pt = p.';

    mu = zeros(t,k);
    z = zeros(t,k);
    nc = (2 * pi())^0.5;

    for i = 1:k
        c_i = c(i);

        mu_i = (indep_s * indep_s_params(:,i)) + (indep_ns * indep_ns_params);
        e_i = dep - mu_i;

        mu(:,i) = mu_i;
        z(:,i) = (1 / (nc * sqrt(c_i))) .* exp(-0.5 .* sum((e_i / c_i) .* e_i,2));
    end

    w1 = (pt * (ones(k,1) .* (1 / k))) .* z(1,:).';
    f1 = ones(1,k) * w1;

    f = [f1; zeros(t - 1,1)];
    g = [(w1 ./ f1).'; zeros(t - 1,k)];

    for i = 2:t
        wi = (pt * g(i-1,:).') .* z(i,:).';
        fi = ones(1,k) * wi;

        f(i,1) = fi;
        g(i,:) = wi ./ fi;
    end

    ll_v = log(f(2:end));

    if (any(~isfinite(ll_v)))
        ll = Inf;
    else
        ll = -sum(ll_v);
    end

end

function [params,lb,ub] = parametrize_indep_ns(dep,indep_ns)

    indep_ns_count = size(indep_ns,2);

    if (indep_ns_count > 0)
        params = regress(dep,indep_ns);
        lb = -Inf(indep_ns_count,1);
        ub = Inf(indep_ns_count,1);
    else
        params = [];
        lb = [];
        ub = [];
    end

end

function [params,lb,ub] = parametrize_indep_s(dep,indep_s,k)

    indep_s_count = size(indep_s,2);

    b = regress(dep,indep_s);
    b_factor = 1;

    indep_s_params = zeros(indep_s_count,k);

    for i = 1:k
        indep_s_params(:,i) = b * b_factor;
        b_factor = b_factor * -1;
    end

    params = reshape(indep_s_params.',indep_s_count * k,1);
    lb = -Inf(numel(params),1);
    ub = Inf(numel(params),1);

end

function [params,lb,ub,ae,be] = parametrize_p(k,tmm,p0)

    tmm_nans = isnan(tmm);

    if (all(all(tmm_nans)))
        params = p0(:);

        lb = zeros(k^2,1);
        ub = ones(k^2,1) - 1e-4;

        ae = repmat(eye(k),1,k);
        be = ones(k,1);
    else
        filter = ~tmm_nans(:);

        params = p0(:);
        params(filter) = [];

        lb = zeros(numel(params),1);
        ub = ones(numel(params),1);

        ae = repmat(eye(k),1,k);
        ae(:,filter) = [];

        be = ones(k,1) -  sum(p0 .* ~tmm_nans,2);
    end

end

function [params,lb,ub] = parametrize_s2(dep,k,vs,factor,multiplier)

    mm = (dep - mean(dep)).^2;
    s2 = var(dep);

    if (vs)
        params = zeros(k,1);

        for i = 1:k
            params(i) = s2 * multiplier;
            multiplier = multiplier * factor;
        end

        lb = ones(k,1) .* min(mm);
        ub = ones(k,1) .* max(mm);
    else
        params = s2 * multiplier;

        lb = min(mm);
        ub = max(mm);
    end

end

function [dep,indep_s,indep_ns,k,finit,tmm,p0,fnlcon] = validate_input(dep,indep_s,indep_ns,k,finit,tmm,fnlcon)

    dep = dep(:);
    t = numel(dep);

    if (~isempty(indep_s))
        if (size(indep_s,1) ~= t)
            error(['The value of ''indep_s'' is invalid. Expected input to have ' num2str(t) ' rows.']);
        end

        if (all(indep_s(:,1) == 1))
            error('The value of ''indep_s'' is invalid. Expected input to exclude the intercept because it is internally handled by the model.');
        end

        if (any(sum(indep_s == 0,1) == t))
            error('The value of ''indep_s'' is invalid. Expected input to contain no zero-valued vectors.');
        end
    end

    indep_s = [ones(t,1) indep_s];

    if (~isempty(indep_ns) && (size(indep_ns,1) ~= t))
        if (size(indep_ns,1) ~= t)
            error(['The value of ''indep_ns'' is invalid. Expected input to have ' num2str(t) ' rows.']);
        end

        if (any(sum(indep_ns,1) == t) || any(sum(indep_ns == 0,1) == t))
            error('The value of ''indep_ns'' is invalid. Expected input to contain no zero-valued or one-valued vectors.');
        end
    end

    if (~isempty(finit))
        if (~isa(finit,'function_handle') || ~isscalar(finit))
            error('The value of ''finit'' is invalid. Expected input to be a single function handle.');
        end

        if (nargin(finit) ~= 8)
            error('The value of ''finit'' is invalid. Expected input to accept 8 input arguments.');
        end

        if (nargout(finit) ~= 7)
            error('The value of ''finit'' is invalid. Expected input to accept 7 output arguments.');
        end
    end

    if (isempty(tmm))
        tmm = NaN(k,k);
        p0 = repmat(0.1,k,k) + (eye(k) * (1 - (k * 0.1)));
    else
        if (any(size(tmm) ~= k))
            error(['The value of ''tmm'' is invalid. Expected input to be a matrix of size ' num2str(k) 'x' num2str(k) '.']);
        end

        tmm_nans = isnan(tmm);
        p0 = zeros(k,k);

        for i = 1:k
            tmm_i = tmm(i,:);
            tmm_nans_i = tmm_nans(i,:);
            tmm_nans_i_count = sum(tmm_nans_i);

            if (tmm_nans_i_count == k)
                p0(i,i) = (1 - (k * 0.1)) + 0.1;

                tmm_nans_i(i) = 0;
                p0(i,tmm_nans_i) = 0.1;
            else
                if (any((tmm_i < 0) | (tmm_i >= 1)))
                    error(['The value of ''tmm'' is invalid. Expected input to have a valid definition for row ' num2str(i) ': all the constrained elements must have a value in range [0,1).']);
                end

                if (tmm_nans_i_count == 0)
                    if (sum(tmm_i) ~= 1)
                        error(['The value of ''tmm'' is invalid. Expected input to have a valid definition for row ' num2str(i) ': the sum of elements must be equal to 1.']);
                    end
                elseif (tmm_nans_i_count == 1)
                    if (k == 2)
                        error(['The value of ''tmm'' is invalid. Expected input to have a valid definition for row ' num2str(i) ': when the number of states is equal to 2, only fully constrained or fully unconstrained rows are accepted.']);
                    else
                        error(['The value of ''tmm'' is invalid. Expected input to have a valid definition for row ' num2str(i) ': the number of unconstrained elements must be greater than 1.']);
                    end
                else
                    if (sum(tmm_i(~tmm_nans_i)) >= 1)
                        error(['The value of ''tmm'' is invalid. Expected input to have a valid definition for row ' num2str(i) ': the number of constrained elements must less than 1.']);
                    end
                end

                if (~tmm_nans_i(i) && (tmm_i(i) == 0))
                    error(['The value of ''tmm'' is invalid. Expected input to have a valid definition for row ' num2str(i) ': diagonal elements cannot be constrained to a value equal to 0.']);
                end

                p0(i,~tmm_nans_i) = tmm_i(~tmm_nans_i);

                k_i = sum(tmm_nans_i);
                delta = 1 - sum(tmm_i(~tmm_nans_i));

                if (tmm_nans_i(i))
                    pd = 0.7 * delta;
                    pnd = (delta - pd) / (k_i - 1);

                    p0(i,i) = pd;

                    tmm_nans_i(i) = 0;
                    p0(i,tmm_nans_i) = pnd;
                else
                    p0(i,tmm_nans_i) = delta / sum(tmm_nans_i);
                end
            end
        end

        we = (k - 1)^2 + 1;
        q = p0^we;
        is_ergodic = ~any(q(:) < (k * eps()));

        if (~is_ergodic)
            error('The value of ''tmm'' is invalid. Expected input to produce an ergodic stochastic row-wise transition matrix.');
        end
    end

    if (~isempty(fnlcon))
        if (~isa(fnlcon,'function_handle') || ~isscalar(fnlcon))
            error('The value of ''fnlcon'' is invalid. Expected input to be a single function handle.');
        end

        if (nargin(fnlcon) ~= 8)
            error('The value of ''fnlcon'' is invalid. Expected input to accept 8 input arguments.');
        end

        if (nargout(fnlcon) ~= 2)
            error('The value of ''fnlcon'' is invalid. Expected input to accept 2 output arguments.');
        end
    end

end
