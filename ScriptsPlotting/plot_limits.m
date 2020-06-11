% [INPUT]
% data = A float t-by-n matrix containing the time series whose plot limits have to be found.
% a = A float [0.00,0.20] representing the adjustment factor for lower and upper limits.
% ll = A float (-Inf,Inf) representing the fixed value for the lower limit (optional, by default the lower limit is retrieved from y).
% ul = A float (-Inf,Inf) representing the fixed value for the upper limit (optional, by default the upper limit is retrieved from y).
% lc = A float (-Inf,Inf) representing the cap for the lower limit (optional, by default no cap is applied to the lower limit).
% uc = A float (-Inf,Inf) representing the cap for the upper limit (optional, by default no cap is applied to the upper limit).
%
% [OUTPUT]
% l = A vector of 2 floats (-Inf,Inf) containing lower and upper limits of y.

function l = plot_limits(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data',@(x)validateattributes(x,{'double'},{'real' '2d' 'nonempty'}));
        ip.addRequired('a',@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.00 '<=' 0.20 'scalar'}));
        ip.addOptional('ll',[],@(x)validateattributes(x,{'double'},{'real' 'finite'}));
        ip.addOptional('ul',[],@(x)validateattributes(x,{'double'},{'real' 'finite'}));
        ip.addOptional('lc',[],@(x)validateattributes(x,{'double'},{'real' 'finite'}));
        ip.addOptional('uc',[],@(x)validateattributes(x,{'double'},{'real' 'finite'}));
    end

    ip.parse(varargin{:});
    
    ipr = ip.Results;
    data = ipr.data;
    a = ipr.a;
    [ll,ul,lc,uc] = validate_input(data,ipr.ll,ipr.ul,ipr.lc,ipr.uc);

    nargoutchk(1,1);

    l = plot_limits_internal(data,a,ll,ul,lc,uc);

end

function l = plot_limits_internal(data,a,ll,ul,lc,uc)

    data = data(:);
    data(~isfinite(data)) = [];

    if (~isempty(ll))
        data_ll = ll;
    else
        if (~isempty(lc))
            y_min = min(min(data),lc);
        else
            y_min = min(data);
        end

        if (y_min < 0)
            data_ll = y_min * (1 + a);
        else
            data_ll = y_min * (1 - a);
        end
    end

    if (~isempty(ul))
        data_ul = ul;
    else
        if (~isempty(uc))
            y_max = max(max(data),uc);
        else
            y_max = max(data);
        end
        
        if (y_max < 0)
            data_ul = y_max * (1 - a);
        else
            data_ul = y_max * (1 + a);
        end
    end
    
    if (data_ll > data_ul)
        data_ll = data_ul;
    end
    
    l = [data_ll data_ul];

end

function [ll,ul,lc,uc] = validate_input(data,ll,ul,lc,uc)
    
    if (all(~isfinite(data(:))))
        error('The value of ''data'' is invalid. Expected input to contain finite values.');
    end

    ll_def = ~isempty(ll);
    
    if (ll_def && ~isscalar(ll))
        error('The value of ''ll'' is invalid. Expected input to be a scalar.');
    end

    ul_def = ~isempty(ul);
    
    if (ul_def && ~isscalar(ul))
        error('The value of ''ul'' is invalid. Expected input to be a scalar.');
    end
    
    if (ll_def && ul_def)
        error('The parameters ''ll'' and ''ul'' cannot be both defined.');
    end
    
    lc_def = ~isempty(lc);
    
    if (lc_def && ~isscalar(lc))
        error('The value of ''lc'' is invalid. Expected input to be a scalar.');
    end
    
    if (ll_def && lc_def)
        error('The parameters ''ll'' and ''lc'' cannot be both defined.');
    end
    
    uc_def = ~isempty(uc);
    
    if (uc_def && ~isscalar(uc))
        error('The value of ''uc'' is invalid. Expected input to be a scalar.');
    end
    
    if (ul_def && uc_def)
        error('The parameters ''ul'' and ''uc'' cannot be both defined.');
    end

end
