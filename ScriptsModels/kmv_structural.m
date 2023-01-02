% [INPUT]
% eq = A vector of floats [0,Inf) of length k representing the market value of equity.
% db = A float or a vector of floats [0,Inf) of length k representing the default barrier.
% r = A float or a vector of floats (-Inf,Inf) of length k representing the annualized risk-free interest rate.
% t = A float or a vector of floats (0,Inf) of length k representing the time to maturity of default barrier.
% op = A string representing the option pricing model used by the Systemic CCA framework (optional, default='BSM'):
%   - 'BSM' for Black-Scholes-Merton;
%   - 'GC' for Gram-Charlier.
%
% [OUTPUT]
% va = A column vector of floats of length k representing the value of assets.
% vap = Output argument representing the distributional parameters of assets whose type depends on the chosen option pricing model:
%   - for Black-Scholes-Merton, a float [0,Inf) representing the annualized volatility of assets;
%   - for Gram-Charlier, a row vector of floats (-Inf,Inf) of length 3 whose values represent respectively the annualized volatility, skewness and excess kurtosis of assets.

function [va,vap] = kmv_structural(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('eq',@(x)validateattributes(x,{'double'},{'real' 'finite' 'nonnegative' 'vector' 'nonempty'}));
        ip.addRequired('db',@(x)validateattributes(x,{'double'},{'real' 'finite' 'nonnegative' 'vector' 'nonempty'}));
        ip.addRequired('r',@(x)validateattributes(x,{'double'},{'real' 'finite' 'vector' 'nonempty'}));
        ip.addRequired('t',@(x)validateattributes(x,{'double'},{'real' 'finite' '>' 0 'vector' 'nonempty'}));
        ip.addRequired('op',@(x)any(validatestring(x,{'BSM' 'GC'})));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    [eq,db,r,t] = validate_input(ipr.eq,ipr.db,ipr.r,ipr.t);
    op = ipr.op;

    nargoutchk(1,2);

    [va,vap] = kmv_structural_internal(eq,db,r,t,op);

end

function [va,vap] = kmv_structural_internal(eq,db,r,t,op)

    df = exp(-r .* t);

    k = numel(r);
    sk = sqrt(k);

    va = eq + (db .* df);
    va_r = diff(log(va));
    va_s = sqrt(252) * std(va_r);

    sst = va_s .* sqrt(t);
    d1 = (log(va ./ db) + ((r + (0.5 * va_s^2)) .* t)) ./ sst;
    d2 = d1 - sst;
    n1 = normcdf(d1);
    n2 = normcdf(d2);

    va_old = va;
    va = eq + ((va .* (1 - n1)) + (db .* df .* n2));

    count = 0;
    error = norm(va - va_old) / sk;

    while ((count < 10000) && (error > 1e-8))
        sst = va_s .* sqrt(t);
        d1 = (log(va ./ db) + ((r + (0.5 * va_s^2)) .* t)) ./ sst;
        d2 = d1 - sst;
        n1 = normcdf(d1);
        n2 = normcdf(d2);

        va_old = va;
        va = eq + ((va .* (1 - n1)) + (db .* df .* n2));
        va_r = diff(log(va));
        va_s = sqrt(252) * std(va_r);

        count = count + 1;
        error = norm(va - va_old) / sk;
    end

    if (strcmp(op,'BSM'))
        vap = va_s;
    else
        va_g = skewness(va_r,0) / sqrt(252);
        va_k = (kurtosis(va_r,0) - 3) / 252;

        vap = [va_s va_g va_k];
    end

end

function [eq,db,r,t] = validate_input(eq,db,r,t)

    eq = eq(:);
    eq_len = numel(eq);

    if (eq_len < 5)
        error('The value of ''eq'' is invalid. Expected input to be a vector containing at least 5 elements.');
    end

    data = {db(:) r(:) t(:)};

    l = unique(cellfun(@numel,data));
    l_scalar = (l == 1);

    if (any(l_scalar))
        if (any(l(~l_scalar) ~= eq_len))
            error(['The number of elements of ''db'', ''r'' and ''t'' must be either 1 or equal to ' num2str(eq_len) '.']);
        end
    else
        if (any(l ~= eq_len))
            error(['The number of elements of ''db'', ''r'' and ''t'' must be either 1 or equal to ' num2str(eq_len) '.']);
        end
    end

    for i = 1:numel(data)
        data_i = data{i};

        if (numel(data_i) == 1)
            data{i} = repmat(data_i,eq_len,1);
        end
    end

    [db,r,t] = deal(data{:});

end
