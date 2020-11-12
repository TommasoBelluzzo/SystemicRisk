% [INPUT]
% data = A float t-by-2 matrix representing the model input; the performed test is aimed to assess whether the first observations are Granger-caused by the second ones.
% a = A float [0.01,0.10] representing the probability level of the F test critical value.
% lag_max = An integer [2,Inf) representing the maximum lag order to be evaluated for both restricted and unrestricted models (optional, default=10).
% lag_sel = A string representing the lag order selection criteria (optional, default='AIC'):
%   - 'AIC' for Akaike's Information Criterion.
%   - 'BIC' for Bayesian Information Criterion.
%   - 'FPE' for Final Prediction Error.
%   - 'HQIC' for Hannan-Quinn Information Criterion.
%
% [OUTPUT]
% f = A float (-Inf,Inf) representing the F test statistic.
% cv = A float (-Inf,Inf) representing the F test critical value.
% h0 = A boolean representing the null hypothesis (the first observation is not Granger-caused by the second one) rejection outcome.
% lag_r = An integer [1,lag_max] representing the selected lag order of the restricted model.
% lag_u = An integer [1,lag_max] representing the selected lag order of the unrestricted model.

function [f,cv,h0,lag_r,lag_u] = granger_causality(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data',@(x)validateattributes(x,{'double'},{'real' 'finite' '2d' 'nonempty' 'size' [NaN 2]}));
        ip.addRequired('a',@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.01 '<=' 0.10 'scalar'}));
        ip.addOptional('lag_max',10,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 1 'scalar'}));
        ip.addOptional('lag_sel','AIC',@(x)any(validatestring(x,{'AIC' 'BIC' 'FPE' 'HQIC'})));
    end

    ip.parse(varargin{:});
    
    ipr = ip.Results;
    [data,lag_max] = validate_input(ipr.data,ipr.lag_max);
    a = ipr.a;
    lag_sel = ipr.lag_sel;

    nargoutchk(3,5);

    [f,cv,h0,lag_r,lag_u] = granger_causality_internal(data,a,lag_max,lag_sel);

end

function [f,cv,h0,lag_r,lag_u] = granger_causality_internal(data,a,lag_max,lag_sel)

    up = isempty(getCurrentTask());

    t = size(data,1);
    x = data(:,1);
    y = data(:,2);

    bt = floor(((lag_max * 2) + 1) / 2);
    x_uni = numel(x);
    y_uni = numel(y);

    if ((x_uni < bt) || (y_uni < bt))
        f = 1;
        cv = 1;
        h0 = true;
        lag_r = 1;
        lag_u = 1;
        
        return;
    end
    
    xb = repmat(x,1,lag_max);
    yb = repmat(y,1,lag_max);
    
    tmp = x;
    tmp(1:lag_max) = NaN;
    tmp = repmat(tmp,1,lag_max);

    crit_r = zeros(lag_max,1);
    k_r = zeros(lag_max,1);
    rss_r = zeros(lag_max,1);

    if (up)
        parfor i = 1:lag_max
            xb_i = xb(:,i);
            tmp_i = tmp(:,i);

            y_star =  tmp_i((i + 1):t);
            x_star = [ones(t - i,1) zeros(t - i,i)];

            for j = 1:i
                x_star(:,j + 1) = xb_i((i - j + 1):(t - j));
            end

            [~,~,e] = regress(y_star,x_star);  
            e(isnan(e)) = []; 

            y_valid = y_star;
            y_valid(isnan(y_star) | any(isnan(x_star),2)) = [];

            d = i + 1;
            k = numel(y_valid);
            sse = e.' * e;

            switch (lag_sel)
                case 'AIC'
                    crit_r(i) = (k * log(sse / k)) + (2 * d);
                case 'BIC'
                    crit_r(i) = (k * log(sse / k)) + (log(k) * d);
                case 'FPE'
                    crit_r(i) = (k * log(sse / k)) + log((k + d + 1) / (k - d - 1));
                otherwise
                    crit_r(i) = (k * log(sse / k)) + (2 * log(log(k)) * d);
            end

            k_r(i) = k;
            rss_r(i) = sse;
        end
    else
        for i = 1:lag_max
            xb_i = xb(:,i);
            tmp_i = tmp(:,i);

            y_star =  tmp_i((i + 1):t);
            x_star = [ones(t - i,1) zeros(t - i,i)];

            for j = 1:i
                x_star(:,j + 1) = xb_i((i - j + 1):(t - j));
            end

            [~,~,e] = regress(y_star,x_star);  
            e(isnan(e)) = []; 

            y_valid = y_star;
            y_valid(isnan(y_star) | any(isnan(x_star),2)) = [];

            d = i + 1;
            k = numel(y_valid);
            sse = e.' * e;

            switch (lag_sel)
                case 'AIC'
                    crit_r(i) = (k * log(sse / k)) + (2 * d);
                case 'BIC'
                    crit_r(i) = (k * log(sse / k)) + (log(k) * d);
                case 'FPE'
                    crit_r(i) = (k * log(sse / k)) + log((k + d + 1) / (k - d - 1));
                otherwise
                    crit_r(i) = (k * log(sse / k)) + (2 * log(log(k)) * d);
            end

            k_r(i) = k;
            rss_r(i) = sse;
        end
    end

    [~,lag_r] = min(crit_r);

    crit_u = zeros(lag_max,1);
    k_u = zeros(lag_max,1);
    rss_u = zeros(lag_max,1);

    if (up)
        parfor i = 1:lag_max
            lag_e = max(i,lag_r);
            fob = lag_e + 1;
            obs = t - lag_e;

            xb_i = xb(:,i);
            yb_i = yb(:,i);
            tmp_i = tmp(:,i);

            y_star = tmp_i(fob:t) ;
            x_star = [ones(obs,1) zeros(obs,lag_r) zeros(obs,i)];

            for j = 1:lag_r
                x_star(:,j + 1) = xb_i((fob - j):((fob - j) + obs - 1));
            end

            for j = 1:i
                x_star(:,j + lag_r + 1) = yb_i((fob - j):((fob - j) + obs - 1));
            end

            [~,~,e] = regress(y_star,x_star);
            e(isnan(e)) = [];

            y_valid = y_star;
            y_valid(isnan(y_star) | any(isnan(x_star),2)) = [];

            d = i + lag_r + 1;
            k = numel(y_valid);
            sse = e.' * e;

            switch (lag_sel)
                case 'AIC'
                    crit_u(i) = (k * log(sse / k)) + (2 * d);
                case 'BIC'
                    crit_u(i) = (k * log(sse / k)) + (log(k) * d);
                case 'FPE'
                    crit_u(i) = (k * log(sse / k)) + log((k + d + 1) / (k - d - 1));
                otherwise
                    crit_u(i) = (k * log(sse / k)) + (2 * log(log(k)) * d);
            end

            k_u(i) = k;
            rss_u(i) = sse;
        end
    else
        for i = 1:lag_max
            lag_e = max(i,lag_r);
            fob = lag_e + 1;
            obs = t - lag_e;

            xb_i = xb(:,i);
            yb_i = yb(:,i);
            tmp_i = tmp(:,i);

            y_star = tmp_i(fob:t) ;
            x_star = [ones(obs,1) zeros(obs,lag_r) zeros(obs,i)];

            for j = 1:lag_r
                x_star(:,j + 1) = xb_i((fob - j):((fob - j) + obs - 1));
            end

            for j = 1:i
                x_star(:,j + lag_r + 1) = yb_i((fob - j):((fob - j) + obs - 1));
            end

            [~,~,e] = regress(y_star,x_star);
            e(isnan(e)) = [];

            y_valid = y_star;
            y_valid(isnan(y_star) | any(isnan(x_star),2)) = [];

            d = i + lag_r + 1;
            k = numel(y_valid);
            sse = e.' * e;

            switch (lag_sel)
                case 'AIC'
                    crit_u(i) = (k * log(sse / k)) + (2 * d);
                case 'BIC'
                    crit_u(i) = (k * log(sse / k)) + (log(k) * d);
                case 'FPE'
                    crit_u(i) = (k * log(sse / k)) + log((k + d + 1) / (k - d - 1));
                otherwise
                    crit_u(i) = (k * log(sse / k)) + (2 * log(log(k)) * d);
            end

            k_u(i) = k;
            rss_u(i) = sse;
        end
    end

    [~,lag_u] = min(crit_u);

    if (k_r(lag_r) == k_u(lag_u))
        f_num = max((rss_r(lag_r) - rss_u(lag_u)) / lag_u,0);
    else
        k_avg = (k_r(lag_r) + k_u(lag_u)) / 2;
        f_num = max((k_avg * ((rss_r(lag_r) / k_r(lag_r)) - (rss_u(lag_u) / k_u(lag_u)))) / lag_u,0);
    end

    df = (k_u(lag_u) - lag_r + lag_u + 1);
    f_den = max(rss_u(lag_u,:) / df,0);

    if ((f_den > 0.0) && (df > 0)) 
        f = f_num / f_den;
        cv = finv(1 - a,lag_u,df);
        h0 = f <= cv;
    else
        f = 1;
        cv = 1;
        h0 = true;
    end

end

function [data,lag_max] = validate_input(data,lag_max)

    t = size(data,1);
    b = (lag_max * 2) + 1;
    
    if (t < 5)
        error('The value of ''data'' is invalid. Expected input to be a matrix with at least 5 rows.');
    end

    if (b >= (t - b))
        error('The number of observations is too low for the specified maximum lag order.');
    end

end
