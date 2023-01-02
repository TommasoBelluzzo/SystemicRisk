% [INPUT]
% data = A float t-by-2 matrix (-Inf,Inf) representing the model input.
% type = A string representing the type of metric to calculate:
%   - 'GG' for Gonzalo-Granger Component Metric;
%   - 'H' for Hasbrouck Information Metric.
% lag_max = An integer [2,t-2] representing the maximum lag order to be evaluated (optional, default=10).
% lag_sel = A string representing the lag order selection criteria (optional, default='AIC'):
%   - 'AIC' for Akaike's Information Criterion;
%   - 'BIC' for Bayesian Information Criterion;
%   - 'FPE' for Final Prediction Error;
%   - 'HQIC' for Hannan-Quinn Information Criterion.
%
% [OUTPUT]
% m1 = A float [0,1] representing the first value of the metric.
% m2 = A float [0,1] representing the second value of the metric.
% lag = An integer [1,lag_max] representing the selected lag order.

function [m1,m2,lag] = price_discovery(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data',@(x)validateattributes(x,{'double'},{'real' 'finite' '2d' 'nonempty' 'size' [NaN 2]}));
        ip.addRequired('type',@(x)any(validatestring(x,{'GG' 'H'})));
        ip.addOptional('lag_max',10,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 1 'scalar'}));
        ip.addOptional('lag_sel','AIC',@(x)any(validatestring(x,{'AIC' 'BIC' 'FPE' 'HQIC'})));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    [data,lag_max] = validate_input(ipr.data,ipr.lag_max);
    type = ipr.type;
    lag_sel = ipr.lag_sel;

    nargoutchk(2,3);

    [m1,m2,lag] = price_discovery_internal(data,type,lag_max,lag_sel);

end

function [m1,m2,lag] = price_discovery_internal(data,type,lag_max,lag_sel)

    t = size(data,1);
    b = t - 2;

    if (lag_max == 1)
        lag = 1;
    else
        lag = select_lag_order(data,lag_max,lag_sel);
    end

    lag_seq = 1:(lag + 1);

    [~,~,e] = regress(data(:,2),[ones(size(data,1),1) data(:,1)]);
    ect = e((lag + 2):end);

    x1_diff = diff(data(:,1));
    x1 = cell2mat(arrayfun(@(i)x1_diff(i:(b - lag + i)),lag_seq,'UniformOutput',false));

    x2_diff = diff(data(:,2));
    x2 = cell2mat(arrayfun(@(i)x2_diff(i:(b - lag + i)),lag_seq,'UniformOutput',false));

    [b1,~,e1] = regress(x1(:,1),[ones(size(ect,1),1) ect x1(:,2:end-1) x2(:,2:end-1)]);
    [b2,~,e2] = regress(x2(:,1),[ones(size(ect,1),1) ect x1(:,2:end-1) x2(:,2:end-1)]);

    d1 = b1(2);
    d2 = b2(2);

    if (strcmp(type,'GG'))
        v = abs(d1) / (abs(d1) + abs(d2));
    else
        s1 = std(e1);
        s2 = std(e2);
        rho = corr(e1,e2);

        vp = ((-d2 * s1) + (d1 * rho * s2)) ^ 2;
        v = vp / (vp + ((d1 * s2 * sqrt(1 - rho^2)) ^ 2));
    end

    m1 = min(max(v,0),1);
    m2 = 1 - m1;

end

function lag = select_lag_order(data,lag_max,lag_sel)

    t = size(data,1);
    lag_sam = lag_max + 1;
    k = t - lag_sam + 1;

    data_lag = zeros(k,lag_sam * 2);

    for i = 1:2
        sv = ((lag_sam - 1) * 2) + i;

        data_i = data(:,i);
        offs = repmat(1:k,1,lag_sam) + repelem(lag_sam:-1:1,1,k) - 1;

        data_lag(:,i:2:sv) = reshape(data_i(offs),[k lag_sam]);
    end

    data_lag = data_lag(:,3:end);
    s = size(data_lag,1);

    data_y = data(lag_sam:end,:);

    ni = 2:2:(2 * lag_max);
    rhs = [ones(k,1) (lag_sam:k+lag_sam-1).'];
    crit = zeros(lag_max,1);

    for i = 1:lag_max
        data_x = [data_lag(:,1:ni(i)) rhs];

        r = zeros(k,2);

        parfor j = 1:2
            [~,~,e] = regress(data_y(:,j),data_x);
            r(:,j) = e;
        end

        cp = zeros(2);

        for cp_i = 1:2
            for cp_j = 1:2
                cp(cp_i,cp_j) = r(:,cp_i).' * r(:,cp_j);
            end
        end

        sigmad = det(cp / k);
        d = (i * 4) + 4;

        switch (lag_sel)
            case 'AIC'
                crit(i) = log(sigmad) + ((2 / s) * d);
            case 'BIC'
                crit(i) = log(sigmad) + ((log(s) / s) * d);
            case 'FPE'
                ns = size(data_x,2);
                crit(i) = ((s + ns) / (s - ns))^2 * sigmad;
            otherwise
                crit(i) = log(sigmad) + (2 * (log(log(s)) / s) * d);
        end
    end

    [~,lag] = min(crit);

end

function [data,lag_max] = validate_input(data,lag_max)

    t = size(data,1);
    b = t - 2;

    if (t < 5)
        error('The value of ''data'' is invalid. Expected input to be a matrix with at least 5 rows.');
    end

    if (lag_max > b)
        error(['The value of ''lag_max'' is invalid. Expected input to be less than or equal to ' num2str(b) '.']);
    end

end
