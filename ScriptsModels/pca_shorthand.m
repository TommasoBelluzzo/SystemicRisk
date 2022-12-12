% [INPUT]
% data = A float t-by-n matrix (-Inf,Inf) representing the model input.
% normalize = A boolean that indicates whether to normalize the model input (optional, default=true).
%
% [OUTPUT]
% coefficients = A float n-by-n matrix (-Inf,Inf) representing the PCA coefficients.
% scores = A float t-by-n matrix (-Inf,Inf) representing the PCA scores.
% explained = A column vector of floats [0,100] of length n representing the percentage of total variance explained by each PCA component.

function [coefficients,scores,explained] = pca_shorthand(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data',@(x)validateattributes(x,{'double'},{'real' '2d' 'nonempty'}));
        ip.addOptional('normalize',true,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    data = validate_input(ipr.data);
    normalize = ipr.normalize;

    nargoutchk(3,3);

    [coefficients,scores,explained] = pca_shorthand_internal(data,normalize);

end

function [coefficients,scores,explained] = pca_shorthand_internal(data,normalize)

    if (normalize)
        for i = 1:size(data,2)
            c = data(:,i);

            m = mean(c,'omitnan');

            s = std(c,'omitnan');
            s(s == 0) = 1;

            data(:,i) = (c - m) ./ s;
        end
    end

    [coefficients,scores,~,~,explained] = pca(data,'Economy',false);

end

function data = validate_input(data)

    nan_indices = any(isnan(data),1);
    data(:,nan_indices) = [];

    [t,n] = size(data);

    if ((t < 5) || (n < 2))
        error('The value of ''data'' is invalid. Expected input to be a matrix with a minimum size of 5x2, after the exclusion of time series containing NaN values.');
    end

end
