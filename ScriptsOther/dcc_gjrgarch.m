% [INPUT]
% data = A float t-by-n matrix containing the model input.
% dcc_q = An integer greater than or equal to 1 representing the lag of the innovation term in the DCC estimator (optional, default=1).
% dcc_p = An integer greater than or equal to 1 representing the lag of the lagged correlation matrices in the DCC estimator (optional, default=1).
% arch_q = Optional argument (default=1) with two possible types:
%   - An integer greater than or equal to 1 representing the lag of the innovation terms in the ARCH estimator.
%   - A vector of integers greater than or equal to 1, of length n, containing the lag of each innovation term in the ARCH estimator.
% garch_p = Optional argument (default=1) with two possible types:
%   - An integer greater than or equal to 1 representing the lag of the innovation terms in the GARCH estimator.
%   - A vector of integers greater than or equal to 1, of length n, containing the lag of each innovation term in the GARCH estimator.
%
% [OUTPUT]
% params = A vector containing the GARCH and DCC parameters.
% p = An n-by-n-by-t matrix of floats containing the conditional correlations.
% h = A t-by-n matrix of floats containing the conditional variances.
% e = A t-by-n matrix of floats containing the standardized residuals.
%
% [NOTES]
% Credit goes to Kevin Sheppard, the author of the original code.

function [params,p,h,e] = dcc_gjrgarch(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data',@(x)validateattributes(x,{'double'},{'real','finite','2d','nonempty'}));
        ip.addOptional('dcc_q',1,@(x)validateattributes(x,{'double'},{'real','finite','integer','>=',1,'scalar'}));
        ip.addOptional('dcc_p',1,@(x)validateattributes(x,{'double'},{'real','finite','integer','>=',1,'scalar'}));
        ip.addOptional('arch_q',1,@(x)validateattributes(x,{'double'},{'real','finite','>=',1,'vector','nonempty'}));
        ip.addOptional('garch_p',1,@(x)validateattributes(x,{'double'},{'real','finite','>=',1,'vector','nonempty'}));
    end

    ip.parse(varargin{:});
    ipr = ip.Results;

    nargoutchk(3,4);

    [params,p,h,e] = dcc_gjrgarch_internal(ipr.data,ipr.dcc_q,ipr.dcc_p,ipr.arch_q,ipr.garch_p);

end

function [params,p,h,e] = dcc_gjrgarch_internal(data,dcc_q,dcc_p,arch_q,garch_p)

    [t,n] = size(data);

    if (length(arch_q) == 1)
        arch_q = ones(1,n) * arch_q;
    end

    if (length(garch_p) == 1)
        garch_p = ones(1,n) * garch_p;
    end
    
    options_lim = 1000 * max(arch_q + garch_p + 1);
	options = optimset(optimset(@fmincon),'Diagnostics','off','Display','off','LargeScale','off');
	options = optimset(options,'MaxFunEvals',options_lim,'MaxIter',options_lim,'MaxSQPIter',1000,'TolFun',1e-6);

    gjr = cell(n,1);
    gjr_params = 0;

    e = zeros(t,n);

    parfor i = 1:n
        data_i = data(:,i);

        [gjr_param,gjr_s] = gjrgarch(data_i,arch_q(i),garch_p(i),options);
        gjr_params = gjr_params + length(gjr_param);
        gjr{i} = struct('Params',gjr_param,'Variance',gjr_s);

        e(:,i) = data_i ./ sqrt(gjr_s);
    end
    
    options = optimset(optimset(@fmincon),'Algorithm','sqp','Diagnostics','off','Display','off','LargeScale','off');
    dcc_params = dcc(dcc_q,dcc_p,e,options);
    
    params = NaN(gjr_params,1);
    params_offset = 1;
    
    h = zeros(t,n);

    for i = 1:n
        gjr_i = gjr{i};
        gjr_i_params = gjr_i.Params;

        params_next = params_offset + length(gjr_i_params);
        params(params_offset:params_next-1,1) = gjr_i_params;
        params_offset = params_next;
        
        h(:,i) = gjr_i.Variance;
    end

    params = [params; dcc_params.'];
    p = dcc_gjrgarch_full(params,data,dcc_q,dcc_p,arch_q,garch_p);

end

function p = dcc_gjrgarch_full(params,data,dcc_q,dcc_p,arch_q,garch_p)

    [t,k] = size(data);
    data_var = var(data(:,1));

    s = zeros(t,k);
    idx = 1;

    for j = 1:k
        p_i = garch_p(j);
        q_i = arch_q(j);
        qq_i = q_i + q_i;
        qqp_i = qq_i + p_i;
        m_i = max(q_i,p_i);

        al_i = params(idx:idx+q_i-1);
        al_i_sum = sum(al_i);
        ga_i = params(idx+q_i:idx+qq_i-1);
        ga_i_sum = sum(ga_i);
        be_i = params(idx+qq_i:idx+qqp_i-1);
        be_i_sum = sum(be_i);

        s_i = zeros(size(data,1),1);
        s_i(1:m_i,1) = std(data(:,j)) ^ 2;

        for i = m_i+1:t
            data_lag = data(i-q_i:i-1,j);
            data_lag_m = data_lag .* data_lag;

            s_i(i,1) = ((1 - (al_i_sum + (0.5 * ga_i_sum) + be_i_sum)) * data_var) + (data_lag_m.' * al_i) + ((data_lag_m .* (data_lag<0)).' * ga_i) + (s_i(i-p_i:i-1,1).' * be_i);
        end

        s(:,j) = s_i;
        idx = idx + qqp_i; 
    end

    rsd = data ./ sqrt(s);

    m = max(dcc_q,dcc_p);
    m1 = m + 1;
    mt = m + t;

    al = params(idx:idx+dcc_q-1);
    be = params(idx+dcc_q:idx+dcc_q+dcc_p-1);
    om = 1 - sum(al) - sum(be);

    q_bar = cov(rsd);
    q_bar_om = q_bar * om;
    q_bar_rep = repmat(q_bar,[1 1 m]);

    rsd = [ones(m,k); rsd];

    pt = zeros(k,k,mt);
    pt(:,:,1:m) = q_bar_rep;
    qt = zeros(k,k,mt);
    qt(:,:,1:m) = q_bar_rep;

    for i = m1:mt
        qt(:,:,i) = q_bar_om;

        for j = 1:dcc_q
            rsd_off = rsd(i-j,:);
            qt(:,:,i) = qt(:,:,i) + (al(j) * (rsd_off.' * rsd_off));
        end

        for j = 1:dcc_p
            qt(:,:,i) = qt(:,:,i) + (be(j) * qt(:,:,i-j));
        end

        qt_i = qt(:,:,i);
        qt_i_sd = sqrt(diag(qt_i));

        pt(:,:,i) = qt_i ./ (qt_i_sd * qt_i_sd.');
    end

    p = pt(:,:,(m1:mt));

end

function params = dcc(dcc_q,dcc_p,e,options)

    m = max(dcc_q,dcc_p);
    m1 = m + 1;

    x0 = [((ones(1,dcc_q) * 0.01) / dcc_q) ((ones(1,dcc_p) * 0.97) / dcc_p)];
    a = ones(size(x0));
    b = 1 - (2 * options.TolCon);
    lb = zeros(size(x0)) + (2 * options.TolCon);

    cache = [m; m1];

    params = fmincon(@dcc_likelihood,x0,a,b,[],[],lb,[],[],options,dcc_q,dcc_p,e,cache);

end

function x = dcc_likelihood(param,dcc_q,dcc_p,e,cache)

    [t,n] = size(e);

    [m,m1] = deal(cache(1),cache(2));
    mt = t + m;

    al = param(1:dcc_q);
    be = param(dcc_q+1:dcc_q+dcc_p);
    om = 1 - sum(al) - sum(be);

    q_bar = cov(e);
    q_bar_om = q_bar * om;

    e = [zeros(m,n); e];

    qt = zeros(n,n,mt);
    qt(:,:,1:m) = repmat(q_bar,[1 1 m]);

    x = 0;

    for i = m1:mt
        e_i = e(i,:);

        qt(:,:,i) = q_bar_om;

        for j = 1:dcc_q
            e_offset = e(i-j,:);
            qt(:,:,i) = qt(:,:,i) + (al(j) * (e_offset.' * e_offset));
        end

        for j = 1:dcc_p
            qt(:,:,i) = qt(:,:,i) + (be(j) * qt(:,:,i-j));
        end

        qt_i = qt(:,:,i);
        qt_i_sd = sqrt(diag(qt_i));
        pt_i = qt_i ./ (qt_i_sd * qt_i_sd.');

        x = x + log(det(pt_i)) + ((e_i / pt_i) * e_i.');
    end

    x = 0.5 * x;

end

function [params,h] = gjrgarch(data,arch_q,garch_p,options)

    q2 = 2 * arch_q;
    qqp = arch_q + arch_q + garch_p;
    
	m = max(garch_p,arch_q);
    m1 = m + 1;

    a = (0.05 * ones(arch_q,1)) / arch_q;
    g = (0.05 * ones(arch_q,1)) / arch_q;
    b = (0.75 * ones(garch_p,1)) / garch_p;
    x0 = [a; g; b];

    ac = [-eye(qqp); ones(1,arch_q) (0.5 * ones(1,arch_q)) ones(1,garch_p)];
    bc = [(zeros(qqp,1) + (2 * options.TolCon)); (1 - (2 * options.TolCon))];
    lb = zeros(1,qqp) + (2 * options.TolCon);     

    data_dev = std(data,1);
    data = [data_dev(ones(m,1)); data];
    data_var = var(data);
    [t,n] = size(data);

    cache = [t; n; data_dev; data_var; q2; m; m1];

    params = fmincon(@gjrgarch_likelihood,x0,ac,bc,[],[],lb,[],[],options,data,arch_q,garch_p,cache);
    [~,h] = gjrgarch_likelihood(params,data,arch_q,garch_p,cache);

end

function [x,h] = gjrgarch_likelihood(params,data,arch_q,garch_p,cache)

    [t,n,data_dev,data_var,q2,m,m1] = deal(cache(1),cache(2),cache(3),cache(4),cache(5),cache(6),cache(7));

    a = params(1:arch_q);
    a_sum = sum(a);
    g = params(arch_q+1:q2);
    g_sum = sum(g);
    b = params(q2+1:q2+garch_p);
    b_sum = sum(b);

    h = zeros(t,n);
    h(1:m,1) = data_dev^2;

    for i = m1:t
        data_lag = data(i-arch_q:i-1);
        data_lag_m = data_lag .* data_lag;

        a_i = data_lag_m.' * a;
        g_i = (data_lag_m .* (data_lag < 0)).' * g;
        b_i = h(i-garch_p:i-1).' * b;

        h(i) = ((1 - (a_sum + (0.5 * g_sum) + b_sum)) * data_var) + a_i + g_i + b_i;
    end

    h = h(m1:t);
    x = 0.5 * ((sum(log(h)) + sum(data(m1:t).^2 ./ h)) + ((t - m) * log(2 * pi)));

end
