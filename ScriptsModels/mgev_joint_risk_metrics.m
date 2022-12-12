% [INPUT]
% l = A float t-by-n matrix [0,Inf) representing the losses.
% k = A float [0.90,0.99] representing the minimum confidence level. Risk indicators are calculated over all the quantiles {0.900;0.925;0.950;0.975;0.990} greater than or equal to k;
%
% [OUTPUT]
% jvars = A row vector of floats [0,Inf) representing the Joint Values-at-Risk.
% jes = A float [0,Inf) representing the Joint Expected Shortfall.

function [jvars,jes] = mgev_joint_risk_metrics(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('l',@(x)validateattributes(x,{'double'},{'real' 'finite' 'nonnegative' '2d' 'nonempty'}));
        ip.addRequired('k',@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.90 '<=' 0.99 'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    l =  validate_input(ipr.l);
    k = ipr.k;

    nargoutchk(2,2);

    [jvars,jes] = mgev_joint_risk_metrics_internal(l,k);

end

function [jvars,jes] = mgev_joint_risk_metrics_internal(l,k)

    persistent options;
    persistent q;

    if (isempty(options))
        options = optimset(optimset(@fmincon),'Algorithm','sqp','Diagnostics','off','Display','off','LargeScale','off');
    end

    if (isempty(q))
        q = [(0.900:0.025:0.975) 0.99];
    end

    up = isempty(getCurrentTask());

    [t,n] = size(l);
    l_sorted = sort(l,1);

    q_fin = q(q >= k);
    q_diff = diff([q_fin 1]);

    xi_s = (1:floor(t / 4)).';
    xi_a = sqrt(log((t - xi_s) ./ t) ./ log(xi_s ./ t));
    xi_q0 = xi_s;
    xi_q1 = floor(t .* (xi_s ./ t).^xi_a);
    xi_q2 = t - xi_s;
    xi_r = (l_sorted(xi_q2,:) - l_sorted(xi_q1,:)) ./ max(1e-8,(l_sorted(xi_q1,:) - l_sorted(xi_q0,:)));

    xi = sum([zeros(1,n); -(log(xi_r) ./ repmat(log(xi_a),1,n))]).' ./ xi_s(end);
    xi_positive = xi > 0;
    xi(xi_positive) = max(0.01,min(2,xi(xi_positive)));
    xi(~xi_positive) = max(-1,min(-0.01,xi(~xi_positive)));

    ms_d = floor(t / 10);
    ms_s = ((ms_d+1):(t-ms_d)).';
    ms_q = -log((1:t).' ./ (t + 1));

    mu = zeros(n,1);
    sigma = zeros(n,1);

    if (up)
        parfor j = 1:n
            y = (ms_q.^-xi(j) - 1) ./ xi(j);
            b = regress(l_sorted(ms_s,j),[ones(numel(ms_s),1) y(ms_s)]);

            mu(j) = b(1);
            sigma(j) = b(2);
        end
    else
        for j = 1:n
            y = (ms_q.^-xi(j) - 1) ./ xi(j);
            b = regress(l_sorted(ms_s,j),[ones(numel(ms_s),1) y(ms_s)]);

            mu(j) = b(1);
            sigma(j) = b(2);
        end
    end

    d_p = tiedrank(l) ./ (t + 1);
    d_y = -1 ./ log(d_p);
    d_v = (d_y ./ repmat(mean(d_y,1),t,1)) ./ (ones(size(l)) .* (1 / n));
    d = min(1,max(1 / mean(min(d_v,[],2)),1 / n));

    x0_mu = n * mean(mu);
    x0_sigma = sqrt(n) * mean(sigma);
    x0_xi = mean(xi);

    jvars = zeros(1,numel(q_fin));

    if (up)
        parfor j = 1:numel(q_fin)
            lhs = -log(q_fin(j)) / d;

            x0 = (x0_mu + (x0_sigma / x0_xi) * (lhs^-x0_xi - 1));
            v0 = (1 + (x0_xi .* ((x0 - x0_mu) ./ x0_sigma))) .^ -(1 ./ x0_xi);

            [jvar,~,ef] = fmincon(@(x)objective(x,v0,lhs,n,mu,sigma,xi),x0,[],[],[],[],0,Inf,[],options);

            if (ef <= 0)
                jvars(j) = 0;
            else
                jvars(j) = jvar;
            end
        end
    else
        for j = 1:numel(q_fin)
            lhs = -log(q_fin(j)) / d;

            x0 = (x0_mu + (x0_sigma / x0_xi) * (lhs^-x0_xi - 1));
            v0 = (1 + (x0_xi .* ((x0 - x0_mu) ./ x0_sigma))) .^ -(1 ./ x0_xi);

            [jvar,~,ef] = fmincon(@(x)objective(x,v0,lhs,n,mu,sigma,xi),x0,[],[],[],[],0,Inf,[],options);

            if (ef <= 0)
                jvars(j) = 0;
            else
                jvars(j) = jvar;
            end
        end
    end

    indices = jvars > 0;

    if (any(indices))
        jes = sum(jvars(indices) .* q_diff(indices)) / sum(q_diff(indices));
    else
        jes = 0;
    end

end

function y = objective(x,v,lhs,n,mu,sigma,xi)

    um = repelem(v,n,1);

    um_check = (xi .* (repelem(x,n,1) - mu)) ./ sigma;
    um_valid = isfinite(um_check) & (um_check > -1);

    x = repelem(x,sum(um_valid),1);
    mu = mu(um_valid);
    sigma = sigma(um_valid);
    xi = xi(um_valid);
    um(um_valid) = (1 + (xi .* ((x - mu) ./ sigma))) .^ -(1 ./ xi);

    y = (sum(um) - lhs)^2;

end

function l = validate_input(l)

    t = size(l,1);

    if (t < 5)
        error('The value of ''l'' is invalid. Expected input to be a matrix with at least 5 rows.');
    end

end
