% [INPUT]
% va = A vector of floats [0,Inf) of length k representing the market value of assets.
% vas = A float [0,Inf) representing the variance of the market value of assets.
% db = A float or a vector of floats [0,Inf) of length k representing the default barrier.
% r = A float or a vector of floats (-Inf,Inf) of length k representing the annualized risk-free interest rate.
% t = A float or a vector of floats (0,Inf) of length k representing the time to maturity of default barrier.
% car = A float [0.03,0.20] representing the capital adequacy ratio used to calculate the Distance to Capital (optional, default=0.08).
%
% [OUTPUT]
% d2d = A column vector of floats (-Inf,Inf) of length k representing the Distance to Default.
% d2c = A column vector of floats (-Inf,Inf) of length k representing the Distance to Capital.

function [d2d,d2c] = default_metrics(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('va',@(x)validateattributes(x,{'double'},{'real' 'finite' 'nonnegative' 'vector' 'nonempty'}));
        ip.addRequired('vas',@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0 'scalar'}));
        ip.addRequired('db',@(x)validateattributes(x,{'double'},{'real' 'finite' 'nonnegative' 'vector' 'nonempty'}));
        ip.addRequired('r',@(x)validateattributes(x,{'double'},{'real' 'finite' 'vector' 'nonempty'}));
        ip.addRequired('t',@(x)validateattributes(x,{'double'},{'real' 'finite' '>' 0 'vector' 'nonempty'}));
        ip.addOptional('car',0.08,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.03 '<=' 0.20 'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    [va,db,r,t] = validate_input(ipr.va,ipr.db,ipr.r,ipr.t);
    vas = ipr.vas;
    car = ipr.car;

    nargoutchk(2,2);

    [d2d,d2c] = default_metrics_internal(va,vas,db,r,t,car);

end

function [d2d,d2c] = default_metrics_internal(va,vas,db,r,t,car)

    rst = (r + (0.5 * vas^2)) .* t;
    st = vas .* sqrt(t);

    d1 = (log(va ./ db) + rst) ./ st;
    d2d = d1 - st;

    d1 = (log(va ./ ((1 / (1 - car)) .* db)) + rst) ./ st;
    d2c = d1 - st;

end

function [va,db,r,t] = validate_input(va,db,r,t)

    va = va(:);
    va_len = numel(va);

    if (va_len < 5)
        error('The value of ''va'' is invalid. Expected input to be a vector containing at least 5 elements.');
    end

    data = {db(:) r(:) t(:)};

    l = unique(cellfun(@numel,data));
    l_scalar = (l == 1);

    if (any(l_scalar))
        if (any(l(~l_scalar) ~= va_len))
            error(['The number of elements of ''db'', ''r'' and ''t'' must be either 1 or equal to ' num2str(va_len) '.']);
        end
    else
        if (any(l ~= va_len))
            error(['The number of elements of ''db'', ''r'' and ''t'' must be either 1 or equal to ' num2str(va_len) '.']);
        end
    end

    for i = 1:numel(data)
        data_i = data{i};

        if (numel(data_i) == 1)
            data{i} = repmat(data_i,va_len,1);
        end
    end

    [db,r,t] = deal(data{:});

end
