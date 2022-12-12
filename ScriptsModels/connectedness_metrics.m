% [INPUT]
% am = A binary n-by-n matrix {0;1} representing the adjcency matrix.
% gd = A vector of integers [1,Inf) of length k representing the group delimiters (optional, default=[]).
%
% [OUTPUT]
% dci = A float [0,Inf) representing the Dynamic Causality Index.
% cio = A float [0,Inf) representing the "In & Out" connections.
% cioo = A float [0,Inf) representing the "In & Out - Other" connections if group delimiters are provided, NaN otherwise.

function [dci,cio,cioo] = connectedness_metrics(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('am',@(x)validateattributes(x,{'double'},{'real' 'finite' 'binary' '2d' 'square' 'nonempty'}));
        ip.addOptional('gd',[],@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' 'positive' 'increasing'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    [am,gd] = validate_input(ipr.am,ipr.gd);

    nargoutchk(2,3);

    [dci,cio,cioo] = connectedness_metrics_internal(am,gd);

end

function [dci,cio,cioo] = connectedness_metrics_internal(am,gd)

    n = size(am,1);

    dci = sum(sum(am)) / ((n ^ 2) - n);

    ni = zeros(n,1);
    no = zeros(n,1);

    for i = 1:n     
        ni(i) = sum(am(:,i));
        no(i) = sum(am(i,:));
    end

    cio = (sum(ni) + sum(no)) / (2 * (n - 1));

    if (isempty(gd))
        cioo = NaN;
    else
        gd_len = length(gd);

        nifo = zeros(n,1);
        noto = zeros(n,1);

        for i = 1:n
            group_1 = gd(1);
            group_n = gd(gd_len);

            if (i <= group_1)
                g_beg = 1;
                g_end = group_1;
            elseif (i > group_n)
                g_beg = group_n + 1;
                g_end = n;
            else
                for j = 1:gd_len-1
                    g_j0 = gd(j);
                    g_j1 = gd(j+1);

                    if ((i > g_j0) && (i <= g_j1))
                        g_beg = g_j0 + 1;
                        g_end = g_j1;
                    end
                end
            end

            nifo(i) = ni(i) - sum(am(g_beg:g_end,i));
            noto(i) = no(i) - sum(am(i,g_beg:g_end));
        end

        cioo = (sum(nifo) + sum(noto)) / (2 * gd_len * (n / gd_len));
    end

end

function [am,gd] = validate_input(am,gd)

    amv = am(:);

    if (numel(amv) < 4)
        error('The value of ''am'' is invalid. Expected input to be a square matrix with a minimum size of 2x2.');
    end

    if (~isempty(gd))
        if (~isvector(gd) || (numel(gd) < 2))
            error('The value of ''gd'' is invalid. Expected input to be a vector containing at least 2 elements.');
        end

        gd = gd(:);
    end

end
