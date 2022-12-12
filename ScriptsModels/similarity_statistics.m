% [INPUT]
% data = A float t-by-2 matrix (-Inf,Inf) representing the model input.
% e = A float (0,2] representing the exponent of the euclidean distance used to calculate the Distance Correlation (optional, default=1).
%
% [OUTPUT]
% dcor = A float [0,1] representing the Distance Correlation.
% rmss = A float [0,1] representing the RMS Similarity.

function [dcor,rmss] = similarity_statistics(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data',@(x)validateattributes(x,{'double'},{'real' 'finite' '2d' 'nonempty' 'size' [NaN 2]}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    data = validate_input(ipr.data);

    nargoutchk(2,2);

    [dcor,rmss] = similarity_statistics_internal(data);

end

function [dcor,rmss] = similarity_statistics_internal(data)

    x1 = data(:,1);
    x2 = data(:,2);

    dcor = calculate_dcor(x1,x2);
    rmss = calculate_rmss(x1,x2);

end

function dcor = calculate_dcor(x1,x2)

    d1 = sqrt(bsxfun(@minus,x1,x1.').^2);
    m1 = mean(d1,1);
    k1 = bsxfun(@minus,bsxfun(@minus,d1,m1.'),m1) + mean(mean(d1));

    d2 = sqrt(bsxfun(@minus,x2,x2.').^2);
    m2 = mean(d2,1);
    k2 = bsxfun(@minus,bsxfun(@minus,d2,m2.'),m2) + mean(mean(d2));

    v1 = sqrt(mean(mean(k1 .* k1)));
    v2 = sqrt(mean(mean(k2 .* k2)));
    v = sqrt(v1 * v2);

    dcov = sqrt(mean(mean(k1 .* k2)));

    if (v > 0)
        dcor = dcov / v;
    else
        dcor = 0;
    end

end

function rmss = calculate_rmss(x1,x2)

    s = 1 - (abs(x1 - x2) ./ (abs(x1) + abs(x2)));
    rmss = sqrt(mean(s.^2,'omitnan'));

end

function data = validate_input(data)

    t = size(data,1);

    if (t < 5)
        error('The value of ''data'' is invalid. Expected input to be a matrix with at least 5 rows.');
    end

end
