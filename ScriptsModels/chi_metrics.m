% [INPUT]
% chi = A float n-by-n-by-t matrix [0,1] representing the time-varying Chi coefficients.
%
% [OUTPUT]
% achi = A row vector of floats [0,1] of length t representing the Average Chi.
% adr = A row vector of floats [0,1] of length t representing the Asymptotic Dependence Rate.

function [achi,adr] = chi_metrics(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('chi',@(x)validateattributes(x,{'double'},{'real' '3d' 'nonempty'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    chi = validate_input(ipr.chi);

    nargoutchk(2,2);

    [achi,adr] = chi_metrics_internal(chi);

end

function [achi,adr] = chi_metrics_internal(chi)

    up = isempty(getCurrentTask());

    [n1,n2,t] = size(chi);
    n = min(n1,n2);

    achi = zeros(t,1);
    adr = zeros(t,1);

    if (up)
        parfor k = 1:t
            chi_k = chi(:,:,k);

            adr_num = 0;
            chi_sum = 0;
            den = 0;

            for i = 1:n
                for j = 1:n
                    if (i == j)
                        continue;
                    end

                    chi_kij = chi_k(i,j);

                    if (isnan(chi_kij))
                        continue;
                    end

                    if (chi_kij > 0)
                        adr_num = adr_num + 1;
                    end

                    chi_sum = chi_sum + chi_kij;
                    den = den + 1;
                end
            end

            achi(k) = chi_sum / den;
            adr(k) = adr_num / den;
        end
    else
        for k = 1:t
            chi_k = chi(:,:,k);

            adr_num = 0;
            chi_sum = 0;
            den = 0;

            for i = 1:n
                for j = 1:n
                    if (i == j)
                        continue;
                    end

                    chi_kij = chi_k(i,j);

                    if (isnan(chi_kij))
                        continue;
                    end

                    if (chi_kij > 0)
                        adr_num = adr_num + 1;
                    end

                    chi_sum = chi_sum + chi_kij;
                    den = den + 1;
                end
            end

            achi(k) = chi_sum / den;
            adr(k) = adr_num / den;
        end
    end

end

function chi = validate_input(chi)

    [n1,n2,t] = size(chi);

    if ((n1 ~= n2) || (min(n1,n2) < 2))
        error('The value of ''chi'' is invalid. Expected input to be a square 3d matrix with a minimum size of 2x2xt.');
    end

    if (t < 5)
        error('The value of ''chi'' is invalid. Expected input to be a square 3d matrix with at least elements on the third dimension.');
    end

    chiv = chi(:);
    chiv(isnan(chiv)) = [];

    if (any((chiv < 0) | (chiv > 1)))
        error('The value of ''chi'' is invalid. Expected input to contain non-NaN values greater than or equal to 0 and less than or equal to 1.');
    end

end
