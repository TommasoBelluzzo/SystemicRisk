% [INPUT]
% y = A float t-by-n matrix containing the time series whose plot limits have to be found.
% a = A float [0.00,0.20] representing the adjustment factor for lower and upper limits.
% ll = A float (-Inf,Inf) representing the fixed value for the lower limit (optional, by default the lower limit is retrieved from y).
% ul = A float (-Inf,Inf) representing the fixed value for the upper limit (optional, by default the upper limit is retrieved from y).
% lc = A float (-Inf,Inf) representing the cap for the lower limit (optional, by default no cap is applied to the lower limit).
% uc = A float (-Inf,Inf) representing the cap for the upper limit (optional, by default no cap is applied to the upper limit).
%
% [OUTPUT]
% l = A vector of 2 floats (-Inf,Inf) containing lower and upper limits of y.

function l = find_plot_limits(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('y',@(x)validateattributes(x,{'double'},{'real','2d','nonempty'}));
        ip.addRequired('a',@(x)validateattributes(x,{'double'},{'real','finite','>=',0.00,'<=',0.20,'scalar'}));
        ip.addOptional('ll',[],@(x)validateattributes(x,{'double'},{'real'}));
        ip.addOptional('ul',[],@(x)validateattributes(x,{'double'},{'real'}));
        ip.addOptional('lc',[],@(x)validateattributes(x,{'double'},{'real'}));
        ip.addOptional('uc',[],@(x)validateattributes(x,{'double'},{'real'}));
    end

    ip.parse(varargin{:});
    
    ipr = ip.Results;
    y = ipr.y;
    a = ipr.a;
    [ll,ul,lc,uc] = validate_input(y,ipr.ll,ipr.ul,ipr.lc,ipr.uc);

    nargoutchk(1,1);

    l = find_plot_limits_internal(y,a,ll,ul,lc,uc);

end

function l = find_plot_limits_internal(y,a,ll,ul,lc,uc)

    y = y(:);
    y(~isfinite(y)) = [];

    if (~isempty(ll))
        y_ll = ll;
    else
        if (~isempty(lc))
            y_min = min(min(y),lc);
        else
            y_min = min(y);
        end

        if (y_min < 0)
            y_ll = y_min * (1 + a);
        else
            y_ll = y_min * (1 - a);
        end
    end

    if (~isempty(ul))
        y_ul = ul;
    else
        if (~isempty(uc))
            y_max = max(max(y),uc);
        else
            y_max = max(y);
        end
        
        if (y_max < 0)
            y_ul = y_max * (1 - a);
        else
            y_ul = y_max * (1 + a);
        end
    end
    
    if (y_ll > y_ul)
        y_ll = y_ul;
    end
    
    l = [y_ll y_ul];

end

function [ll,ul,lc,uc] = validate_input(y,ll,ul,lc,uc)
    
    if (all(~isfinite(y(:))))
        error('The value of ''y'' is invalid. Expected input to contain finite values.');
    end

    ll_def = ~isempty(ll);
    ul_def = ~isempty(ul);
    
    if (ll_def && ul_def)
        error('The parameters ''ll'' and ''ul'' cannot be both defined.');
    end
    
    lc_def = ~isempty(lc);
    
    if (ll_def && lc_def)
        error('The parameters ''ll'' and ''lc'' cannot be both defined.');
    end
    
    uc_def = ~isempty(uc);
    
    if (ul_def && uc_def)
        error('The parameters ''ul'' and ''uc'' cannot be both defined.');
    end

    if (ll_def)
        if (~isscalar(ll))
            error('The value of ''ll'' is invalid. Expected input to be a scalar.');
        end

        if (~isfinite(ll))
            error('The value of ''ll'' is invalid. Expected input to be finite.');
        end
    end

    if (ul_def)
        if (~isscalar(ul))
            error('The value of ''ul'' is invalid. Expected input to be a scalar.');
        end

        if (~isfinite(ul))
            error('The value of ''ul'' is invalid. Expected input to be finite.');
        end
    end
    
    if (lc_def)
        if (~isscalar(lc))
            error('The value of ''lc'' is invalid. Expected input to be a scalar.');
        end

        if (~isfinite(lc))
            error('The value of ''lc'' is invalid. Expected input to be finite.');
        end
    end

    if (uc_def)
        if (~isscalar(uc))
            error('The value of ''uc'' is invalid. Expected input to be a scalar.');
        end

        if (~isfinite(uc))
            error('The value of ''uc'' is invalid. Expected input to be finite.');
        end
    end

end
