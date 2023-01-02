% [INPUT]
% pods = A vector of floats [0,1] of length n representing the probabilities of default.
% g = A boolean n^2-by-n matrix representing the posterior density orthants.
% p = A vector of floats [0,1] of length n^2 representing the posterior density probabilities.
%
% [OUTPUT]
% jpod = A float [0,1] representing the Joint Probability of Default.
% fsi = A float [1,n] representing the Financial Stability Index.
% pce = A float [0,1] representing the Probability of Cascade Effects.
% dide = A float n-by-n matrix [0,1] representing the Distress Dependency.
% si = A row vector of floats [0,1] of length n representing the Systemic Importance.
% sv = A row vector of floats [0,1] of length n representing the Systemic Vulnerability.
% cojpods = A row vector of floats [0,1] of length n representing the Conditional Joint Probabilities of Default.

function [jpod,fsi,pce,dide,si,sv,cojpods] = cross_entropy_metrics(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('pods',@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0 '<=' 1 'vector' 'nonempty'}));
        ip.addRequired('g',@(x)validateattributes(x,{'double'},{'real' 'finite' 'binary' '2d' 'nonempty'}));
        ip.addRequired('p',@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0 '<=' 1 'vector' 'nonempty'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    [pods,g,p] = validate_input(ipr.pods,ipr.g,ipr.p);

    nargoutchk(7,7);

    [jpod,fsi,pce,dide,si,sv,cojpods] = cross_entropy_metrics_internal(pods,g,p);

end

function [jpod,fsi,pce,dide,si,sv,cojpods] = cross_entropy_metrics_internal(pods,g,p)

    n = numel(pods);
    g_refs = sum(g,2);

    jpod = p(g_refs == n,:);
    fsi = min(max(sum(pods,'omitnan') / (1 - p(g_refs == 0,:)),1),n);
    pce = sum(p(g_refs >= 2,:)) / sum(p(g_refs >= 1,:));

    dide = eye(n);

    for i = 1:n
        for j = 1:n
            if (isnan(pods(j)))
                dide(i,j) = NaN;
            elseif (i ~= j)
                dide(i,j) = p((g_refs == 2) & (g(:,i) == 1) & (g(:,j) == 1),:) / pods(j);
            end
        end
    end

    dide_pods = ((dide - eye(n)) .* repmat(pods,1,n));
    si = sum(dide_pods,2);
    sv = sum(dide_pods,1).';

    jpods = ones(n,1) .* jpod;
    cojpods = (jpods ./ pods).';

end

function [pods,g,p] = validate_input(pods,g,p)

    pods = pods(:);
    n = numel(pods);

    if (n < 2)
        error('The value of ''pods'' is invalid. Expected input to be a vector containing at least 2 elements.');
    end

    k = n^2;

    [kg,ng] = size(g);

    if ((kg ~= k) || (ng ~= n))
        error(['The value of ''g'' is invalid. Expected input to be a matrix of size ' num2str(k) 'x' num2str(n) '.']);
    end

    kp = numel(p);

    if (kp ~= k)
        error(['The value of ''p'' is invalid. Expected input to be a vector containing ' num2str(k) ' elements.']);
    end

    p = p(:);

end
