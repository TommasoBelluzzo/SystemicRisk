% [INPUT]
% m = A variable representing the target measures, with two possible types:
%   - A cell array of length k, where k is the number of measures, of t-by-n matrices of numerical values (-Inf,Inf).
%   - A t-by-n-by-k matrix of numerical values (-Inf,Inf), where k is the number of measures.
%
% [OUTPUT]
% rc = A column vector of floats of length n representing the value of assets.
% rs = A column vector of floats of length n representing the value of assets.

function [rc,rs] = kendall_rankings(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('m',@(x)validateattributes(x,{'cell' 'double'},{'nonempty'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    m = validate_input(ipr.m);

    nargoutchk(2,2);

    [rc,rs] = kendall_rankings_internal(m);

end

function [rc,rs] = kendall_rankings_internal(m)

    [t,~,k] = size(m);
    p = nchoosek(1:k,2);

    rc = zeros(k,k);
    rs = zeros(1,k);

    for i = 1:size(p,1)
        p_i = p(i,:);

        index_1 = p_i(1);
        measure_1 = m(:,:,index_1);

        index_2 = p_i(2);
        measure_2 = m(:,:,index_2);

        for j = 1:t
            [~,rank_1] = sort(measure_1(j,:),'ascend');
            [~,rank_2] = sort(measure_2(j,:),'ascend');

            rc(index_1,index_2) = rc(index_1,index_2) + concordance_coefficient(rank_1.',rank_2.');
        end
    end

    for i = 1:k
        measure = m(:,:,i);

        for j = t:-1:2
            [~,rank_previous] = sort(measure(j-1,:),'ascend');
            [~,rank_current] = sort(measure(j,:),'ascend');

            rs(i) = rs(i) + concordance_coefficient(rank_current.',rank_previous.');
        end
    end

    rc = ((rc + rc.') / t) + eye(k);
    rs = rs ./ (t - 1);

end

function cc = concordance_coefficient(rank_1,rank_2)

    m = [rank_1 rank_2];
    [g,f] = size(m);

    rm = zeros(g,f);

    for i = 1:f
        x_i = m(:,i);
        [~,b] = sort(x_i);
        rm(b,i) = 1:g;
    end

    rm_sum = sum(rm,2);
    s = sum(rm_sum.^2,1) - (sum(rm_sum)^2 / g);

    cc = (12 * s) / ((f ^ 2) * (g^3 - g));

end

function m_final = validate_input(m)

    if (iscell(m))
        if (~isvector(m))
            error('The value of ''m'' is invalid. Expected input to be a vector, when specified as a cell array.');
        end

        k = numel(m);

        if (k < 2)
            error('The value of ''m'' is invalid. Expected input to contain at least 2 elements, when specified as a cell array.');
        end

        ts = zeros(k,1);
        ns = zeros(k,1);

        for i = 1:k
            m_i = m{i};

            if (~ismatrix(m_i))
                error('The value of ''m'' is invalid. Expected input to contain only matrices, when specified as a cell array.');
            end

            [t,n] = size(m_i);

            if ((t < 5) || (n < 2))
                error('The value of ''m'' is invalid. Expected input to contain matrices with a minimum size of 5x2.');
            end

            ts(i) = t;
            ns(i) = n;
        end

        ts_uni = unique(ts);
        ns_uni = unique(ns);

        if ((numel(ts_uni) ~= 1) || (numel(ns_uni) ~= 1))
            error('The value of ''m'' is invalid. Expected input to contain equally sized matrices, when specified as a cell array.');
        end

        t = ts_uni(1);
        n = ns_uni(1);
        m_final = zeros(t,n,k);

        for i = 1:k
            m_final(:,:,i) = m{i};
        end
    else
        m_size = size(m);

        if (numel(m_size) ~= 3)
            error('The value of ''m'' is invalid. Expected input to have 3 dimensions, when specified as a matrix.');
        end

        [t,n,k] = deal(m_size(1),m_size(2),m_size(3));

        if (k < 2)
            error('The value of ''m'' is invalid. Expected input to have third dimension greater than or equal to 2, when specified as a matrix.');
        end

        if ((t < 5) || (n < 2))
            error('The value of ''m'' is invalid. Expected input to have a minimum 2-dimensional size of 5x2.');
        end

        m_final = m;
    end

end
