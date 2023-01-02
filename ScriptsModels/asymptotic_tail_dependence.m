% [INPUT]
% data = A float t-by-n matrix (-Inf,Inf) representing the model input.
% tail = A string representing the target tail:
%   - 'L' for lower tail;
%   - 'U' for upper tail.
% bw = An integer [21,252] representing the dimension of each rolling window (optional, default=252).
% f = A float [0.05,0.20] representing the percentage of observations to be included in tails (optional, default=0.10).
% pt = A float [0,1) representing the initial penantly term for underrepresented samples with respect to the bandwidth (optional, default=0.5).
%
% [OUTPUT]
% chi = A float n-by-n-by-t matrix [0,1] representing the Chi coefficients.
% chi_bar = A float n-by-n-by-t matrix [-1,1] representing the Chi Bar coefficients.
%
% [NOTES]
% The bandwidth is automatically expanded as much as possible following an optimality criteria.

function [chi,chi_bar] = asymptotic_tail_dependence(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data',@(x)validateattributes(x,{'double'},{'real' '2d' 'nonempty'}));
        ip.addRequired('tail',@(x)any(validatestring(x,{'L' 'U'})));
        ip.addOptional('bw',252,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 21 '<=' 252 'scalar'}));
        ip.addOptional('f',0.10,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.05 '<=' 0.2 'scalar'}));
        ip.addOptional('pt',0.5,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0 '<' 1 'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    [data,bw] = validate_input(ipr.data,ipr.bw);
    tail = ipr.tail;
    f = ipr.f;
    pt = ipr.pt;

    nargoutchk(1,2);

    [chi,chi_bar] = asymptotic_tail_dependence_internal(data,bw,tail,f,pt);

end

function [chi,chi_bar] = asymptotic_tail_dependence_internal(data,bw,tail,f,pt)

    up = isempty(getCurrentTask());

    [t,n] = size(data);

    c = nchoosek(1:n,2);
    c_len = size(c,1);

    dc = cell(c_len,2);

    for i = 1:c_len
        c_i = c(i,:);
        dc(i,:) = {c_i data(:,c_i)};
    end

    pt = [linspace(pt,1,bw).'; ones(t-bw,1)];

    dc_results = cell(c_len,2);

    if (up)
        parfor k = 1:c_len
            windows = extract_rolling_windows(dc{k,2},bw);

            chi_k = zeros(t,1);
            chibar_k = zeros(t,1);

            for w = 1:t
                dc_kw = windows{w};

                [chi_kw,chibar_kw] = calculate_coefficients(dc_kw,tail,f);
                chi_k(w) = chi_kw;
                chibar_k(w) = chibar_kw;
            end

            dc_results(k,:) = {(chi_k .* pt) (chibar_k .* pt)};
        end
    else
        for k = 1:c_len
            windows = extract_rolling_windows(dc{k,2},bw);

            chi_k = zeros(t,1);
            chibar_k = zeros(t,1);

            for w = 1:t
                dc_kw = windows{w};

                [chi_kw,chibar_kw] = calculate_coefficients(dc_kw,tail,f);
                chi_k(w) = chi_kw;
                chibar_k(w) = chibar_kw;
            end

            dc_results(k,:) = {(chi_k .* pt) (chibar_k .* pt)};
        end
    end

    en = eye(n);
    chi = repmat(en,1,1,t);
    chi_bar = repmat(en,1,1,t);

    for k = 1:c_len
        dc_k = dc{k,1};
        i = dc_k(1);
        j = dc_k(2);

        [chi_k,chibar_k] = deal(dc_results{k,:});
        chi(i,j,:) = chi_k;
        chi(j,i,:) = chi_k;
        chi_bar(i,j,:) = chibar_k;
        chi_bar(j,i,:) = chibar_k;
    end

    for i = 1:t
        nan_indices = isnan(data(i,:));

        if (any(nan_indices))
            chi(nan_indices,:) = NaN;
            chi(:,nan_indices) = NaN;

            chi_bar(nan_indices,:) = NaN;
            chi_bar(:,nan_indices) = NaN;
        end
    end

end

function [chi,chi_bar] = calculate_coefficients(data,tail,f)

    if (any(any(isnan(data),1)))
        chi = NaN;
        chi_bar = NaN;
        return;
    end

    t = size(data,1);
    t1 = t + 1;

    nu = max(round(t * f),1);
    nu1 = nu + 1;

    if (strcmp(tail,'L'))
        u1 = 1 - (tiedrank(data(:,1)) ./ t1);
        u2 = 1 - (tiedrank(data(:,2)) ./ t1);
    else
        u1 = tiedrank(data(:,1)) ./ t1;
        u2 = tiedrank(data(:,2)) ./ t1;
    end

    zs = -1 ./ log(u1);
    zt = -1 ./ log(u2);
    z =  sort(min(zs,zt),'descend');

    if (nu == 1)
        eta = 0;
    else
        eta = sum(log(z(1:nu) ./ z(nu1)));
    end

    chi_bar = ((2 / nu1) * eta) - 1;
    sigma = (chi_bar + 1)^2 / nu1;
    ci = chi_bar + (norminv(0.975) * sigma^0.5);

    if (ci >= 1)
        chi = z(nu1) * (nu1 / t);
    else
       chi = 0;
    end

end

function [data,bw] = validate_input(data,bw)

    [t,n] = size(data);

    if (n < 2)
        error('The value of ''data'' is invalid. Expected input to be a matrix with at least 2 columns.');
    end

    nan_counts = sum(isnan(data),1);
    nan_threshold = round(t * 0.70,0);

    if (any(nan_counts > nan_threshold))
        error(['The value of ''data'' is invalid. Expected input to contain no more than 70% of NaN values (' num2str(nan_threshold) ') for each time series.']);
    end

    for i = 2:ceil(t / bw)
        bw_i = bw * i;

        if ((bw_i / t) >= 0.3)
            bw = bw_i;
            break;
        end
    end

end
