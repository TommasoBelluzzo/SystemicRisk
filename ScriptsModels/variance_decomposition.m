% [INPUT]
% data = A float t-by-n matrix (-Inf,Inf) representing the model input.
% fevd = A string representing the FEVD type:
%   - 'G' for generalized FEVD;
%   - 'O' for orthogonal FEVD.
% lags = An integer [1,3] representing the number of lags of the VAR model (optional, default=2).
% h = An integer [1,10] representing the prediction horizon (optional, default=4).

%
% [OUTPUT]
% vd = A float n-by-n matrix (-Inf,Inf) representing the variance decomposition.

function vd = variance_decomposition(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data',@(x)validateattributes(x,{'double'},{'real' '2d' 'nonempty'}));
        ip.addRequired('fevd',@(x)any(validatestring(x,{'G' 'O'})));
        ip.addOptional('lags',2,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 1 '<=' 3 'scalar'}));
        ip.addOptional('h',4,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 1 '<=' 10 'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    data = validate_input(ipr.data);
    fevd = ipr.fevd;
    lags = ipr.lags;
    h = ipr.h;

    nargoutchk(1,1);

    vd = variance_decomposition_internal(data,fevd,lags,h);

end

function vd = variance_decomposition_internal(data,fevd,lags,h) 

    [t,n] = size(data);
    d = max(n * 5,t) - t;
    k = t + d - lags;

    if (d > 0)
        mu = repmat(mean(data,1),d,1);
        sigma = repmat(std(data,1),d,1);

        c = corr(data);
        c(isnan(c)) = 0;
        c = nearest_spd(c);

        z = (normrnd(mu,sigma,[d n]) * chol(c,'upper')) + (0.01 .* randn(d,n));

        data = [data; z];
    end

    nan_indices = isnan(data);
    data(nan_indices) = 0;

    zero_indices = find(~data);
    data(zero_indices) = (-9e-9 .* rand(numel(zero_indices),1)) + 1e-8;

    novar_indices = find(var(data,1) == 0);
    data(:,novar_indices) = data(:,novar_indices) + ((-9e-9 .* rand(size(data(:,novar_indices)))) + 1e-8);

    e = [data(1:lags,:); data(lags+1:end,:)];

    ar_first = n + 1;
    ar_start = (lags * n^2) + ar_first;
    trend = ar_start:(ar_start+n-1);

    params = (lags * n^2) + (2 * n);
    f = NaN(params,1);
    f(trend) = zeros(n,1);
    fs = true(params,1);
    fs(trend) = false;

    z = zeros(n,params);
    z(:,1:n) = eye(n);

    x = cell(k,1);
    y = e(lags+1:end,:);

    for t = (lags + 1):(lags + k)
        ar_start = ar_first;
        ar_x = t - lags;

        for i = 1:lags
            indices = ar_start:(ar_start + n^2 - 1);
            z(:,indices) = kron(e(t-i,:),eye(n));
            ar_start = indices(end) + 1;
        end

        z(:,trend) = ar_x * eye(n);
        x{ar_x} = z(:,fs);
        y(ar_x,:) = y(ar_x,:) - (z(:,~fs) * f(~fs)).';
    end

    [b,c] = mvregress(x,y,'CovType','full','VarFormat','beta','VarType','fisher');
    f(fs) = b;

    coefficients = cell(1,lags);
    ar_start = ar_first;

    for i = 1:lags
        indices = ar_start:(ar_start + n^2 - 1);
        coefficients{i} = reshape(f(indices),n,n);
        ar_start = indices(end) + 1;
    end

    g = zeros(n * lags,n * lags);
    g(1:n,:) = cell2mat(coefficients);

    if (lags > 2)
        g(n+1:end,1:(end-n)) = eye((lags - 1) * n);
    end

    ma = cell(h,1);
    ma{1} = eye(n);
    ma{2} = g(1:n,1:n);

    if (h >= 3)
        for i = 3:h
            temp = g^i;
            ma{i} = temp(1:n,1:n);
        end
    end

    irf = zeros(h,n,n);
    vds = zeros(h,n,n);

    if (strcmp(fevd,'G'))
        sigma = diag(c);

        for i = 1:n
            indices = zeros(n,1);
            indices(i,1) = 1;

            for j = 1:h
                irf(j,:,i) = (sigma(i,1) .^ -0.5) .* (ma{j} * c * indices);
            end
        end
    else
        c = nearest_spd(c);
        cl = chol(c,'lower');

        for i = 1:n
            indices = zeros(n,1);
            indices(i,1) = 1;

            for j = 1:h
                irf(j,:,i) = ma{j} * cl * indices; 
            end
        end
    end

    irf_cs = cumsum(irf.^2);
    irf_cs_sum = sum(irf_cs,3);

    for i = 1:n
        vds(:,:,i) = irf_cs(:,:,i) ./ irf_cs_sum;     
    end

    vd = squeeze(vds(h,:,:));
    vd = vd ./ repmat(sum(vd,2),1,n);

end

function c_hat = nearest_spd(c)

    a = (c + c.') ./ 2;
    [~,s,v] = svd(a);
    h = v * s * v.';

    c_hat = (a + h) ./ 2;
    c_hat = (c_hat + c_hat.') ./ 2;

    k = 0;
    p = 1;

    while (p ~= 0)
        [~,p] = chol(c_hat,'upper');
        k = k + 1;

        if (p ~= 0)
            d = min(eig(c_hat));
            c_hat = c_hat + (((-d * k^2) + eps(d)) * eye(size(c)));
        end
    end

end

function data = validate_input(data)

    k = numel(data);
    k_nan = sum(sum(isnan(data)));

    if ((k_nan / k) > 0.25)
        error('The value of ''data'' is invalid. Expected input to contain a number of NaN values not greater 25% of the total number of elements.');
    end

end
