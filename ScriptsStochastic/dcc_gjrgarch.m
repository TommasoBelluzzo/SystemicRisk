% [INPUT]
% data = A numeric t-by-n matrix containing the model input.
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
% p = An n-by-n-by-t matrix of floats containing the DCC coefficients.
% s = A t-by-n matrix of floats containing the conditional variances.
%
% [NOTES]
% Credit goes to Kevin Sheppard, the author of the original code.

function [p,s] = dcc_gjrgarch(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data',@(x)validateattributes(x,{'numeric'},{'2d','nonempty','real','finite'}));
        ip.addOptional('dcc_q',1,@(x)validateattributes(x,{'numeric'},{'scalar','integer','real','finite','>=',1}));
        ip.addOptional('dcc_p',1,@(x)validateattributes(x,{'numeric'},{'scalar','integer','real','finite','>=',1}));
        ip.addOptional('arch_q',1,@(x)validateattributes(x,{'numeric'},{'vector','nonempty','real','finite','>=',1}));
        ip.addOptional('garch_p',1,@(x)validateattributes(x,{'numeric'},{'vector','nonempty','real','finite','>=',1}));
    end

    ip.parse(varargin{:});
    ipr = ip.Results;

    nargoutchk(2,2);

    [p,s] = dcc_gjrgarch_internal(ipr.data,ipr.dcc_q,ipr.dcc_p,ipr.arch_q,ipr.garch_p);

end

function [p,s] = dcc_gjrgarch_internal(data,dcc_q,dcc_p,arch_q,garch_p)

    [t,k] = size(data);

    if (length(arch_q) == 1)
        arch_q = ones(1,k) * arch_q;
    end

    if (length(garch_p) == 1)
        garch_p = ones(1,k) * garch_p;
    end

    opt_lim = 1000 * max(arch_q + garch_p + 1);
    options = optimset('fmincon');
    options = optimset(options,'Display','off');
    options = optimset(options,'Diagnostics','off');
    options = optimset(options,'LargeScale','off');
    options = optimset(options,'MaxFunEvals',opt_lim);
    options = optimset(options,'MaxIter',opt_lim);  
    options = optimset(options,'MaxSQPIter',1000);
    options = optimset(options,'TolFun',1e-006);

    gjr = cell(k,1);
    gjr_params = 0;

    rsd = zeros(t,k);

    parfor i = 1:k
        data_i = data(:,i);

        [gjr_param,gjr_s] = gjrgarch(data_i,arch_q(i),garch_p(i),options);

        gjr_params = gjr_params + length(gjr_param);
        
        gjr{i} = struct('param',gjr_param,'s',gjr_s);
        rsd(:,i) = data_i ./ sqrt(gjr_s);
    end

    options = optimset('fmincon');
    options = optimset(options,'Algorithm','sqp');
    options = optimset(options,'Display','off');
    options = optimset(options,'Diagnostics','off');
    options = optimset(options,'LargeScale','off');

    dcc_param = dcc(dcc_q,dcc_p,rsd,options);
    
    param = NaN(gjr_params,1);
    param_off = 1;
    
    s = zeros(t,k);

    for i = 1:k
        gjr_i = gjr{i};
        gjr_i_param = gjr_i.param;

        param2_off_nxt = param_off + length(gjr_i_param);
        param(param_off:param2_off_nxt-1,1) = gjr_i_param;
        param_off = param2_off_nxt;
        
        s(:,i) = gjr_i.s;
    end

    p = dcc_gjrgarch_fulllikelihood([param; dcc_param'],data,dcc_q,dcc_p,arch_q,garch_p);

end

function p = dcc_gjrgarch_fulllikelihood(param,data,dcc_q,dcc_p,arch_q,garch_p)

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

        al_i = param(idx:idx+q_i-1);
        al_i_sum = sum(al_i);
        ga_i = param(idx+q_i:idx+qq_i-1);
        ga_i_sum = sum(ga_i);
        be_i = param(idx+qq_i:idx+qqp_i-1);
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

    al = param(idx:idx+dcc_q-1);
    be = param(idx+dcc_q:idx+dcc_q+dcc_p-1);
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

function param = dcc(dcc_q,dcc_p,rsd,opt)

    m = max(dcc_q,dcc_p);
    m1 = m + 1;
    opt_2tc = 2 * opt.TolCon;

    x0 = [((ones(1,dcc_q) * 0.01) / dcc_q) ((ones(1,dcc_p) * 0.97) / dcc_p)];
    a = ones(size(x0));
    b = 1 - opt_2tc;
    a_eq = [];
    b_eq = [];
    lb = zeros(size(x0)) + opt_2tc;
    ub = [];
    nlc = [];

    cache = [m; m1];

    param = fmincon(@dcc_likelihood,x0,a,b,a_eq,b_eq,lb,ub,nlc,opt,dcc_q,dcc_p,rsd,cache);

end

function x = dcc_likelihood(param,dcc_q,dcc_p,rsd,cache)

    [t,k] = size(rsd);
    m = cache(1);
    m1 = cache(2);
    mt = t + m;

    al = param(1:dcc_q);
    be = param(dcc_q+1:dcc_q+dcc_p);
    om = 1 - sum(al) - sum(be);

    q_bar = cov(rsd);
    q_bar_om = q_bar * om;

    rsd = [zeros(m,k); rsd];

    qt = zeros(k,k,mt);
    qt(:,:,1:m) = repmat(q_bar,[1 1 m]);

    x = 0;

    for i = m1:mt
        rsd_i = rsd(i,:);

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
        pt_i = qt_i ./ (qt_i_sd * qt_i_sd.');

        x = x + log(det(pt_i)) + ((rsd_i / pt_i) * rsd_i.');
    end

    x = 0.5 * x;

end

function [param,s] = gjrgarch(data,arch_q,garch_p,opt)

    q2 = 2 * arch_q;
    qqp = arch_q + arch_q + garch_p;
    m = max(garch_p,arch_q);
    m1 = m + 1;
    opt_2tc = 2 * opt.TolCon;

    al = (0.05 * ones(arch_q,1)) / arch_q;
    ga = (0.05 * ones(arch_q,1)) / arch_q;
    be = (0.75 * ones(garch_p,1)) / garch_p;

    x0 = [al; ga; be];
    a = [-eye(qqp); ones(1,arch_q) (0.5 * ones(1,arch_q)) ones(1,garch_p)];
    b = [(zeros(qqp,1) + opt_2tc); (1 - opt_2tc)];
    a_eq = [];
    b_eq = [];
    lb = zeros(1,qqp) + opt_2tc;     
    ub = [];
    nlc = [];

    data_dev = std(data,1);
    data = [data_dev(ones(m,1)); data];
    data_var = var(data);
    [t,k] = size(data);

    cache = [t; k; data_dev; data_var; q2; m; m1];

    param = fmincon(@gjrgarch_likelihood,x0,a,b,a_eq,b_eq,lb,ub,nlc,opt,data,arch_q,garch_p,cache);
    [~,s] = gjrgarch_likelihood(param,data,arch_q,garch_p,cache);

end

function [x,s] = gjrgarch_likelihood(param,data,arch_q,garch_p,cache)

    t = cache(1);
    k = cache(2);
    data_dev = cache(3);
    data_var = cache(4);
    q2 = cache(5);
    m = cache(6);
    m1 = cache(7);

    al = param(1:arch_q);
    al_sum = sum(al);
    ga = param(arch_q+1:q2);
    ga_sum = sum(ga);
    be = param(q2+1:q2+garch_p);
    be_sum = sum(be);

    s = zeros(t,k);
    s(1:m,1) = data_dev ^ 2;

    for i = m1:t
        data_lag = data(i-arch_q:i-1);
        data_lag_m = data_lag .* data_lag;

        al_i = data_lag_m.' * al;
        ga_i = (data_lag_m .* (data_lag<0)).' * ga;
        be_i = s(i-garch_p:i-1).' * be;

        s(i) = ((1 - (al_sum + (0.5 * ga_sum) + be_sum)) * data_var) + al_i + ga_i + be_i;
    end

    s = s(m1:t);
    x = 0.5 * ((sum(log(s)) + sum((data(m1:t) .^ 2) ./ s)) + ((t - m) * log(2 * pi)));

end
