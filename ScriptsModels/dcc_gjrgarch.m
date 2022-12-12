% [INPUT]
% data = A float t-by-n matrix (-Inf,Inf) representing the model input.
% dcc_q = An integer [1,Inf) representing the lag of the innovation term in the DCC estimator (optional, default=1).
% dcc_p = An integer [1,Inf) representing the lag of correlation matrices in the DCC estimator (optional, default=1).
% arch_q = Optional argument (default=1) with two possible types:
%   - An integer greater than or equal to 1 representing the lag of the innovation terms in the ARCH estimator.
%   - A vector of integers [1,Inf) of length n representing the lag of each innovation term in the ARCH estimator.
% garch_p = Optional argument (default=1) with two possible types:
%   - An integer greater than or equal to 1 representing the lag of the innovation terms in the GARCH estimator.
%   - A vector of integers [1,Inf) of length n representing the lag of each innovation term in the GARCH estimator.
%
% [OUTPUT]
% p = A float n-by-n-by-t matrix [-1,1] representing the conditional correlations.
% h = A float t-by-n matrix [0,Inf) representing the conditional variances.
% e = A float t-by-n matrix (-Inf,Inf) representing the standardized residuals.
% dcc_params = A row vector of floats (-Inf,Inf) of length 2 representing the DCC parameters.
% gjr_params = An n-by-1 cell array of float vectors (-Inf,Inf) containing the GARCH parameters.
%
% [NOTES]
% Credit goes to Kevin Sheppard, the author of the original code.

function [p,h,e,dcc_params,gjr_params] = dcc_gjrgarch(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data',@(x)validateattributes(x,{'double'},{'real' 'finite' '2d' 'nonempty'}));
        ip.addOptional('dcc_q',1,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 1 'scalar'}));
        ip.addOptional('dcc_p',1,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 1 'scalar'}));
        ip.addOptional('arch_q',1,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 1 'vector' 'nonempty'}));
        ip.addOptional('garch_p',1,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 1 'vector' 'nonempty'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    data = ipr.data;
    dcc_q = ipr.dcc_q;
    dcc_p = ipr.dcc_p;
    arch_q = ipr.arch_q;
    garch_p = ipr.garch_p;

    nargoutchk(2,5);

    [p,h,e,dcc_params,gjr_params] = dcc_gjrgarch_internal(data,dcc_q,dcc_p,arch_q,garch_p);

end

function [p,h,e,dcc_params,gjr_params] = dcc_gjrgarch_internal(data,dcc_q,dcc_p,arch_q,garch_p)

    persistent dcc_options;
    persistent gjr_options;

    if (isempty(dcc_options))
        dcc_options = optimset(optimset(@fmincon),'Algorithm','sqp','Diagnostics','off','Display','off','LargeScale','off');
    end

    if (isempty(gjr_options))
        gjr_options = optimset(optimset(@fmincon),'Algorithm','sqp','Diagnostics','off','Display','off','LargeScale','off','MaxSQPIter',1000,'TolFun',1e-6);
    end

    up = isempty(getCurrentTask());

    [t,n] = size(data);

    for i = 1:n
        data_i = data(:,i);
        data(:,i) = data_i - mean(data_i);
    end

    if (numel(arch_q) == 1)
        arch_q = ones(1,n) .* arch_q;
    end

    if (numel(garch_p) == 1)
        garch_p = ones(1,n) .* garch_p;
    end

    options_lim = 1000 * max(arch_q + garch_p + 1);
    options = optimset(gjr_options,'MaxFunEvals',options_lim,'MaxIter',options_lim);

    gjr_params = cell(n,1);
    h = zeros(t,n);
    e = zeros(t,n);

    if (up)
        parfor i = 1:n
            data_i = data(:,i);

            [gjr_p,gjr_h] = gjrgarch(data_i,arch_q(i),garch_p(i),options);
            gjr_params{i} = gjr_p;

            h(:,i) = gjr_h;
            e(:,i) = data_i ./ sqrt(gjr_h);
        end
    else
        for i = 1:n
            data_i = data(:,i);

            [gjr_p,gjr_h] = gjrgarch(data_i,arch_q(i),garch_p(i),options);
            gjr_params{i} = gjr_p;

            h(:,i) = gjr_h;
            e(:,i) = data_i ./ sqrt(gjr_h);
        end
    end

    dcc_params = dcc(e,dcc_q,dcc_p,dcc_options);

    p = dcc_gjrgarch_core(data,dcc_q,dcc_p,arch_q,garch_p,gjr_params,dcc_params,up);

end

function p = dcc_gjrgarch_core(data,dcc_q,dcc_p,arch_q,garch_p,gjr_params,dcc_params,up)

    [t,k] = size(data);
    data_s2 = var(data(:,1));

    s = zeros(t,k);

    if (up)
        parfor i = 1:k
            p_i = garch_p(i);
            q_i = arch_q(i);

            q2_i = q_i * 2;

            m_i = max(q_i,p_i);

            gjr_p = gjr_params{i};
            a_i = gjr_p(1:q_i);
            g_i = gjr_p(q_i+1:q2_i);
            b_i = gjr_p(q2_i+1:end);

            s_i = zeros(size(data,1),1);
            s_i(1:m_i,1) = std(data(:,i)) ^ 2;

            for j = m_i+1:t
                data_l = data(j-q_i:j-1,i);
                data_l2 = data_l.^2;

                s1 = (1 - (sum(a_i) + (0.5 * sum(g_i)) + sum(b_i))) * data_s2;
                s2 = data_l2.' * a_i;
                s3 = (data_l2 .* (data_l < 0)).' * g_i;
                s4 = s_i(j-p_i:j-1,1).' * b_i;
                s_i(j,1) = s1 + s2 + s3 + s4;
            end

            s(:,i) = s_i;
        end
    else
        for i = 1:k
            p_i = garch_p(i);
            q_i = arch_q(i);

            q2_i = q_i * 2;

            m_i = max(q_i,p_i);

            gjr_p = gjr_params{i};
            a_i = gjr_p(1:q_i);
            g_i = gjr_p(q_i+1:q2_i);
            b_i = gjr_p(q2_i+1:end);

            s_i = zeros(size(data,1),1);
            s_i(1:m_i,1) = std(data(:,i)) ^ 2;

            for j = m_i+1:t
                data_l = data(j-q_i:j-1,i);
                data_l2 = data_l.^2;

                s1 = (1 - (sum(a_i) + (0.5 * sum(g_i)) + sum(b_i))) * data_s2;
                s2 = data_l2.' * a_i;
                s3 = (data_l2 .* (data_l < 0)).' * g_i;
                s4 = s_i(j-p_i:j-1,1).' * b_i;
                s_i(j,1) = s1 + s2 + s3 + s4;
            end

            s(:,i) = s_i;
        end
    end

    e = data ./ sqrt(s);

    m = max(dcc_q,dcc_p);
    m1 = m + 1;
    mt = m + t;

    a = dcc_params(1:1+dcc_q-1);
    b = dcc_params(1+dcc_q:1+dcc_q+dcc_p-1);
    o = 1 - sum(a) - sum(b);

    q_bar = cov(e);
    q_bar_o = q_bar * o;
    q_bar_r = repmat(q_bar,[1 1 m]);

    e = [ones(m,k); e];

    pt = zeros(k,k,mt);
    pt(:,:,1:m) = q_bar_r;

    qt = zeros(k,k,mt);
    qt(:,:,1:m) = q_bar_r;

    for i = m1:mt
        qt(:,:,i) = q_bar_o;

        for j = 1:dcc_q
            e_ij = e(i-j,:);
            qt(:,:,i) = qt(:,:,i) + (a(j) * (e_ij.' * e_ij));
        end

        for j = 1:dcc_p
            qt(:,:,i) = qt(:,:,i) + (b(j) * qt(:,:,i-j));
        end

        qt_i = qt(:,:,i);
        qt_i_sd = sqrt(diag(qt_i));

        pt(:,:,i) = qt_i ./ (qt_i_sd * qt_i_sd.');
    end

    p = pt(:,:,(m1:mt));

end

function params = dcc(e,q,p,options)

    [t,n] = size(e);

    qpm = max(q,p);
    qpmt = t + qpm;

    qp = q + p;

    tol = 2 * options.TolCon;

    x0 = [((ones(1,q) .* 0.01) ./ q) ((ones(1,p) .* 0.97) ./ p)];
    ai = ones(1,qp);
    bi = 1 - tol;
    lb = zeros(1,qp) + tol;

    params = fmincon(@(x)dcc_likelihood(x,e,n,q,p,qpm,qpmt,qp),x0,ai,bi,[],[],lb,[],[],options);

    function ll = dcc_likelihood(x,e,n,q,p,qpm,qpmt,qp)

        a = x(1:q);
        b = x(q+1:qp);
        o = 1 - sum(a) - sum(b);

        q_bar = cov(e);
        q_bar_o = q_bar * o;

        e = [zeros(qpm,n); e];

        qt = zeros(n,n,qpmt);
        qt(:,:,1:qpm) = repmat(q_bar,[1 1 qpm]);

        ll = 0;

        for i = qpm+1:qpmt
            e_i = e(i,:);

            qt_i = q_bar_o;

            for j = 1:q
                e_offset = e(i-j,:);
                qt_i = qt_i + (a(j) * (e_offset.' * e_offset));
            end

            for j = 1:p
                qt_i = qt_i + (b(j) * qt(:,:,i-j));
            end

            qt(:,:,i) = qt_i;
            qt_i_sd = sqrt(diag(qt_i));

            pt_i = qt_i ./ (qt_i_sd * qt_i_sd.');

            ll = ll + log(det(pt_i)) + ((e_i / pt_i) * e_i.');
        end

        ll = 0.5 * ll;

    end

end

function [params,h] = gjrgarch(data,q,p,options)

    qpm = max(p,q);
    qpm1 = qpm + 1;

    q2 = q * 2;
    q2p = q2 + p;

    tol = 2 * options.TolCon;

    x0 = [(ones(q,1) .* (0.05 / q)); (ones(q,1) .* (0.10 / q)); ((ones(p,1) .* 0.75) ./ p)];
    ai = [-eye(q2p); ones(1,q) (ones(1,q) .* 0.5) ones(1,p)];
    bi = [(ones(q2p,1) .* -tol); (1 - tol)];
    lb = ones(1,q2p) .* tol;

    s = std(data,1);
    data = [s(ones(max(p,q),1)); data];
    s2 = var(data);

    params = fmincon(@(x)gjrgarch_likelihood(x,data,s,s2,q,p,qpm,qpm1,q2),x0,ai,bi,[],[],lb,[],[],options);
    [~,h] = gjrgarch_likelihood(params,data,s,s2,q,p,qpm,qpm1,q2);

    function [ll,h] = gjrgarch_likelihood(x,data,s,s2,q,p,qpm,qpm1,q2)

        [t,n] = size(data);

        a = x(1:q);
        g = x(q+1:q2);
        b = x(q2+1:end);

        a_sum = sum(a);
        g_sum = 0.5 * sum(g);
        b_sum = sum(b);

        h = zeros(t,n);
        h(1:qpm,1) = s^2;

        for i = qpm1:t
            data_l = data(i-q:i-1);
            data_l2 = data_l.^2;

            a_i = data_l2.' * a;
            g_i = (data_l2 .* (data_l < 0)).' * g;
            b_i = h(i-p:i-1).' * b;

            h(i) = ((1 - (a_sum + g_sum + b_sum)) * s2) + a_i + g_i + b_i;
        end

        h = h(qpm1:t);

        ll = 0.5 * ((sum(log(h)) + sum(data(qpm1:t).^2 ./ h)) + ((t - qpm) * log(2 * pi())));

    end

end
