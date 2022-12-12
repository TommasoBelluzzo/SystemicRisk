% [INPUT]
% va = A vector of floats [0,Inf) of length n representing the market values of assets.
% vap = Input argument representing the distributional parameters of assets whose type depends on the chosen option pricing model:
%   - for Black-Scholes-Merton, a float [0,Inf) representing the annualized volatility of assets;
%   - for Gram-Charlier, a row vector of floats (-Inf,Inf) of length 3 whose values represent respectively the annualized volatility, skewness and excess kurtosis of assets.
% cds = A vector of floats [0,Inf) of length n representing the credit default swap spreads.
% db = A float or a vector of floats [0,Inf) of length n representing the default barrier.
% r = A float or a vector of floats (-Inf,Inf) of length n representing the annualized risk-free interest rate.
% t = A float or a vector of floats (0,Inf) of length n representing the time to maturity of default barrier.
%
% [OUTPUT]
% el = A column vector of floats [0,Inf) of length n representing the expected losses.
% cl = A column vector of floats [0,Inf) of length n representing the contingent liabilities.
% a = A column vector of floats [0,1] of length n representing the contingent alphas.

function [el,cl,a] = contingent_claims_analysis(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('va',@(x)validateattributes(x,{'double'},{'real' 'finite' 'nonnegative' '2d' 'nonempty'}));
        ip.addRequired('vap',@(x)validateattributes(x,{'double'},{'real' 'finite' 'vector' 'nonempty'}));
        ip.addRequired('cds',@(x)validateattributes(x,{'double'},{'real' 'nonnegative' 'vector' 'nonempty'}));
        ip.addRequired('db',@(x)validateattributes(x,{'double'},{'real' 'finite' 'nonnegative' 'vector' 'nonempty'}));
        ip.addRequired('r',@(x)validateattributes(x,{'double'},{'real' 'finite' 'vector' 'nonempty'}));
        ip.addRequired('t',@(x)validateattributes(x,{'double'},{'real' 'finite' '>' 0 'vector' 'nonempty'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    [va,vap,cds,db,r,t] = validate_input(ipr.va,ipr.vap,ipr.cds,ipr.db,ipr.r,ipr.t);

    nargoutchk(3,3);

    [el,cl,a] = contingent_claims_analysis_internal(va,vap,cds,db,r,t);

end

function [el,cl,a] = contingent_claims_analysis_internal(va,vap,cds,db,r,t)

    s = vap(1);
    st = s * sqrt(t);

    dbd = db .* exp(-r .* t);

    d1 = (log(va ./ db) + ((r + (0.5 * s^2)) .* t)) ./ st;
    d2 = d1 - st;

    put_price = (dbd .* normcdf(-d2)) - (va .* normcdf(-d1));

    if (numel(vap) == 3)
        g = vap(2);
        k = vap(3);

        t1 = (g / 6) .* ((2 * s) - d1);
        t2 = (k / 24) .* (1 - d1.^2 + (3 .* d1 .* s) - (3 * s^2));

        put_price = put_price - (va .* normcdf(d1) .* s .* (t1 - t2));
    end

    put_price = max(0,put_price);

    rd = dbd - put_price;

    cds_put_price = dbd .* (1 - exp(-cds .* max(0.5,((db ./ rd) - 1)) .* t));
    cds_put_price = min(cds_put_price,put_price);  

    a = max(0,min(1 - (cds_put_price ./ put_price),1));
    a(~isreal(a)) = 0;

    el = put_price;
    cl = el .* a;

end

function [va,vap,cds,db,r,t] = validate_input(va,vap,cds,db,r,t)

    va = va(:);
    va_len = numel(va);

    if (va_len < 5)
        error('The value of ''va'' is invalid. Expected input to be a vector containing at least 5 elements.');
    end

    vap_len = numel(vap);

    if ((vap_len ~= 1) && (vap_len ~= 3))
        error('The value of ''vap'' is invalid. Expected input to be a vector containing either 1 or 3 elements.');
    end

    if (vap(1) < 0)
        error('The value of ''vap'' is invalid. Expected input first element to be greater than or equal to 0.');
    end

    cds = cds(:);

    if (numel(va) ~= va_len)
        error(['The value of ''cds'' is invalid. Expected input to be a vector of length ' num2str(va_len) ' elements.']);
    end

    if (all(cds >= 1))
        cds = cds ./ 10000;
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
