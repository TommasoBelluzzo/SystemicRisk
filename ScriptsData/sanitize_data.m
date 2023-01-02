% [INPUT]
% data = A float t-by-n matrix containing the time series to be sanitized.
% x = A vector of length t containing the numeric observation dates of the time series (if empty, observations are assumed to be linearly spaced between 1 and t).
% w = An integer [5,21] representing the length of the moving window used to detect outliers (if empty, no outliers replacement is performed).
% m = A vector of 2 floats (-Inf,Inf) containing minimum and maximum clamping values (if empty, no clamping is performed).
%
% [OUTPUT]
% data = A float t-by-n matrix containing the sanitized time series.

function data = sanitize_data(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data',@(x)validateattributes(x,{'double'},{'real' '2d' 'nonempty'}));
        ip.addRequired('x',@(x)validateattributes(x,{'double'},{'real'}));
        ip.addRequired('w',@(x)validateattributes(x,{'double'},{'real'}));
        ip.addRequired('m',@(x)validateattributes(x,{'double'},{'real'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    data = ipr.data;
    [x,w,m] = validate_input(data,ipr.x,ipr.w,ipr.m);

    nargoutchk(1,1);

    data = sanitize_data_internal(data,x,w,m);

end

function data = sanitize_data_internal(data,x,w,m)

    for i = 1:size(data,2)
        y = data(:,i);

        nan_indices = isnan(y);

        if (all(nan_indices))
            continue;
        end

        if (any(nan_indices))
            y = fill_missing_values(y,x,nan_indices);
        end

        if (~isempty(w))
            y = replace_outliers(y,x,w);
        end

        if (~isempty(m))
            y = min(max(y,m(1)),m(2));
        end

        data(:,i) = y;
    end

end

function y = fill_missing_values(y,x,nan_indices)

    d = diff(nan_indices);

    if (nan_indices(1))
        z = find(d == -1,1,'first');
        y(1:z) = y(z+1);
        nan_indices(1:z) = 0;
    end

    if (nan_indices(end))
        z = find(d == 1,1,'last');
        y(z+1:end) = y(z);
        nan_indices(z+1:end) = 0;
    end

    y(nan_indices) = spline(x(~nan_indices),y(~nan_indices),x(nan_indices));

end

function y = replace_outliers(y,x,w)

    f = 3 * (-1 / (sqrt(2) * erfcinv(1.5)));
    k = floor(w * 0.5);
    n = numel(y);
    p = nan(k,1);

    xp = [p; y; p];

    m_med = zeros(n,1);
    m_mad = zeros(n,1);

    for i = 1:k
        x_i = y(1:k+i);
        m = median(x_i);

        m_med(i) = m;
        m_mad(i) = median(abs(x_i - m));
    end

    for i = k+1:n-k-1
        x_i = xp(i:i+w,:);
        m = median(x_i);

        m_med(i) = m;
        m_mad(i) = median(abs(x_i - m));
    end

    for i = n-k:n
        x_i = y(i-k:end);
        m = median(x_i);

        m_med(i) = m;
        m_mad(i) = median(abs(x_i - m));
    end

    b = m_mad .* f;
    lb = m_med - b;
    ub = m_med + b;

    is_outlier = (y > ub) | (y < lb);
    d = diff(is_outlier);

    if (is_outlier(1) == 1)
        z = find(d == -1,1,'first');
        y(1:z) = y(z+1);
        is_outlier(1:z) = 0;
    end

    if (is_outlier(end))
        z = find(d == 1,1,'last');
        y(z+1:end) = y(z);
        is_outlier(z+1:end) = 0;
    end

    y(is_outlier) = spline(x(~is_outlier),y(~is_outlier),x(is_outlier));

end

function [x,w,m] = validate_input(data,x,w,m)

    t = size(data,1);

    if (isempty(x))
        x = 1:t;
    else
        if (~isvector(x))
            error('The value of ''x'' is invalid. Expected input to be a vector.');
        end

        if (numel(x) ~= t)
            error(['The value of ''x'' is invalid. Expected input to contain ' num2str(t) ' elements.']);
        end

        if (~all(isfinite(x)))
            error('The value of ''x'' is invalid. Expected input to contain finite elements.');
        end

        if (~all(diff(x) > 0))
            error('The value of ''x'' is invalid. Expected input to contain increasing elements.');
        end
    end

    if (~isempty(w))
        if (~isscalar(w))
            error('The value of ''w'' is invalid. Expected input to be a scalar.');
        end

        if (~isfinite(w))
            error('The value of ''w'' is invalid. Expected input to be finite.');
        end

        if (floor(w) ~= w)
            error('The value of ''w'' is invalid. Expected input to be an integer.');
        end

        if ((w < 5) || (w > 21))
            error('The value of ''w'' is invalid. Expected input to have a value >= 5 and <= 21.');
        end

        w = w - 1;
    end

    if (~isempty(m))
        if (~isvector(m))
            error('The value of ''m'' is invalid. Expected input to be a vector.');
        end

        if (numel(m) ~= 2)
            error('The value of ''m'' is invalid. Expected input to contain 2 elements.');
        end

        if (~all(isfinite(m)))
            error('The value of ''m'' is invalid. Expected input to contain finite elements.');
        end

        if (any(floor(m) ~= m))
            error('The value of ''m'' is invalid. Expected input to contain integer elements.');
        end

        if (m(1) >= m(2))
            error('The value of ''m'' is invalid. Expected input first element to be less than the input second element.');
        end
    end

end
