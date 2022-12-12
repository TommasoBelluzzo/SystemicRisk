% [INPUT]
% data = A float t-by-n matrix (-Inf,Inf) representing the model input.
% a = A float [0.01,0.10] representing the target quantile.
% k = An integer [1,60] representing the target lag.
% cis = A float (0.0,0.1] representing the significance level of confidence intervals (optional, default=0.050).
% cib = An integer [10,1000] representing the number of bootstrap iterations of confidence intervals (optional, default=100).
%
% [OUTPUT]
% cq = A float (-Inf,Inf) representing the cross-quantilogram.
% ci = A row vector of floats (-Inf,Inf) of length 2 representing the lower and upper confidence intervals.
%
% [NOTES]
% The model computes partial cross-quantilograms when n is greater than 2 using exogenous variables from 3 to n.

function [cq,ci] = cross_quantilograms_sb(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data',@(x)validateattributes(x,{'double'},{'real' 'finite' '2d' 'nonempty'}));
        ip.addRequired('a',@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.01 '<=' 0.10 'scalar'}));
        ip.addRequired('k',@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 1 '<=' 60 'scalar'}));
        ip.addOptional('cis',0.050,@(x)validateattributes(x,{'double'},{'real' 'finite' '>' 0 '<=' 0.1 'scalar'}));
        ip.addOptional('cib',100,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 10 '<=' 1000 'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    data = validate_input(ipr.data);
    a = ipr.a;
    k = ipr.k;
    cis = ipr.cis;
    cib = ipr.cib;

    nargoutchk(2,2);

    [cq,ci] = cross_quantilograms_sb_internal(data,a,k,cis,cib);

end

function [cq,ci] = cross_quantilograms_sb_internal(data,a,k,cis,cib)

    [t,n] = size(data);
    len = t - k;
    partial = n > 2;

    cis = cis / 2;

    d = zeros(len,n);
    d(:,1) = data(k+1:t,1);
    d(:,2:n) = data(1:len,2:n);

    block_length = ppw_optimal_block_length(d);
    g = mean(block_length(:,1));

    a_sb = ones(len,n) .* a;
    cq_sb = zeros(cib,1);

    if (partial)
        for i = 1:cib
            indices = indices_bootstrap(len,g);

            d_sb = d(indices,:);
            q_sb = (d_sb <= repmat(gumbel_quantile(d_sb,a),len,1)) - a_sb;

            h_sb = q_sb.' * q_sb;

            if (det(h_sb) <= 1e-08)
                hi_sb = pinv(h_sb);
            else
                hi_sb = inv(h_sb);
            end

            cq_sb(i) = -hi_sb(1,2) / sqrt(hi_sb(1,1) * hi_sb(2,2));
        end
    else
        for i = 1:cib
            indices = indices_bootstrap(len,g);

            d_sb = d(indices,:);
            q_sb = (d_sb <= repmat(gumbel_quantile(d_sb,a),len,1)) - a_sb;

            h_sb = q_sb.' * q_sb;

            cq_sb(i) = h_sb(1,2) / sqrt(h_sb(1,1) * h_sb(2,2));
        end
    end

    q = (data <= repmat(gumbel_quantile(data,a),t,1)) - (ones(t,n) .* a);

    d = zeros(len,n);
    d(:,1) = q(k+1:t,1);
    d(:,2:n) = q(1:len,2:n);

    h = d.' * d;

    if (partial)
        if (det(h) <= 1e-08)
            hi = pinv(h);
        else
            hi = inv(h);
        end

        cq = -hi(1,2) / sqrt(hi(1,1) * hi(2,2));
    else
        cq = h(1,2) / sqrt(h(1,1) * h(2,2));
    end

    cqc = cq_sb - cq;
    ci = [min(0,gumbel_quantile(cqc,cis)) max(0,gumbel_quantile(cqc,1 - cis))];

end

function q = gumbel_quantile(x,p)

    index = 1 + ((size(x,1) - 1) * p);
    low = floor(index);
    high = ceil(index);

    x = sort(x);
    x_low = x(low,:);
    x_high = x(high,:);

    h = max(index - low,0);
    q = (h .* x_high) + ((1 - h) .* x_low);

end

function indices = indices_bootstrap(n,g)

    indices = [ceil(n * rand()); zeros(n - 1,1)];

    u = rand(n,1) < g;
    indices(u) = ceil(n .* rand(sum(u),1));

    zi = find(~u(2:n));
    indices(zi + 1) = indices(zi) + 1;

    fi = indices > n;
    indices(fi) = indices(fi) - n;

end

function bl = ppw_optimal_block_length(x)

    [t,n] = size(x);

    k = max(sqrt(log10(t)),5);
    c = 2 * sqrt(log10(t) / t);

    b_max = ceil(min(3 * sqrt(t),t / 3));
    m_max = ceil(sqrt(t)) + k;

    bl = zeros(n,2);

    for ppw_i = 1:n
        x_i = x(:,ppw_i);

        p1 = m_lag(x_i,m_max);
        p1 = p1(m_max+1:end,:);
        p1 = corr([x_i(m_max+1:end) p1]);
        p1 = p1(2:end,1);

        p2 = [m_lag(p1,k).' p1(end-k+1:end)];
        p2 = p2(:,k+1:end);
        p2 = sum((abs(p2) < (ones(k,m_max - k + 1) .* c))).';

        p3 = [(1:length(p2)).' p2];
        p3 = p3(p2 == k,:);

        if (isempty(p3))
            m_hat = find(abs(p1) > c,1,'last');
        else
            m_hat = p3(1,1);
        end

        m = min(2 * m_hat,m_max);

        if (m > 0)
            mm = (-m:m).';

            p1 = m_lag(x_i,m);
            p1 = p1(m+1:end,:);
            p1 = cov([x_i(m+1:end),p1]);

            act = sortrows([-(1:m).' p1(2:end,1)],1);
            ac = [act(:,2); p1(:,1)];

            mmn = mm ./ m;
            kernel_weights = ((abs(mmn) >= 0) .* (abs(mmn) < 0.5)) + (2 .* (1 - abs(mmn)) .* (abs(mmn) >= 0.5) .* (abs(mmn) <= 1));

            acw = kernel_weights .* ac;
            acw_ss = sum(acw)^2;

            g_hat = sum(acw .* abs(mm));
            dcb_hat = (4/3) * acw_ss;
            dsb_hat = 2 * acw_ss;

            b_comp1 = 2 * g_hat^2;
            b_comp2 = t^(1 / 3);
            bl_vl = min((b_comp1 / dsb_hat)^(1/3) * b_comp2,b_max);
            bl_cb = min((b_comp1 / dcb_hat)^(1/3) * b_comp2,b_max);

            bl(ppw_i,:) = [bl_vl bl_cb];
        else
            bl(ppw_i,:) = 1;
        end
    end

    function l = m_lag(x,n)

        mn = numel(x);
        l = ones(mn,n);

        for ml_i = 1:n
            l(ml_i+1:mn,ml_i) = x(1:mn-ml_i,1);
        end

    end

end

function data = validate_input(data)

    n = size(data,2);

    if (n < 2)
        error('The value of ''data'' is invalid. Expected input to be a matrix with at least 2 columns.');
    end

end
