% [INPUT]
% data = A float t-by-n matrix (-Inf,Inf) representing the model input.
% a = A float [0.01,0.10] representing the target quantile.
% k = An integer [1,60] representing the target lag.
% cis = A float {0.005;0.010;0.025;0.050;0.100} representing the significance level of confidence intervals (optional, default=0.050).
% cif = A float {0.00;0.01;0.03;0.05;0.10;0.15;0.20;0.30} representing the minimum subsample size, as a fraction, for confidence intervals (optional, default=0.10).
%
% [OUTPUT]
% cq = A float (-Inf,Inf) representing the cross-quantilogram.
% ci = A row vector of floats (-Inf,Inf) of length 2 representing the lower and upper confidence intervals.
%
% [NOTES]
% The model computes partial cross-quantilograms when n is greater than 2 using exogenous variables from 3 to n.

function [cq,ci] = cross_quantilograms_sn(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data',@(x)validateattributes(x,{'double'},{'real' 'finite' '2d' 'nonempty'}));
        ip.addRequired('a',@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.01 '<=' 0.10 'scalar'}));
        ip.addRequired('k',@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 1 '<=' 60 'scalar'}));
        ip.addOptional('cis',0.050,@(x)validateattributes(x,{'double'},{'real' 'finite' 'scalar'}));
        ip.addOptional('cif',0.10,@(x)validateattributes(x,{'double'},{'real' 'finite' 'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    data = validate_input_data(ipr.data);
    a = ipr.a;
    k = ipr.k;
    [civ,cif] = validate_input_ci(ipr.cis,ipr.cif);

    nargoutchk(2,2);

    [cq,ci] = cross_quantilograms_sn_internal(data,a,k,civ,cif);

end

function [cq,ci] = cross_quantilograms_sn_internal(x,a,k,civ,cif)

    [t,n] = size(x);
    len = max(round(cif * t,0),1);
    partial = n > 2;

    cq_sn = zeros(t,1);

    if (partial)
        for i = len:t
            x_i = x(1:i,:);
            x_t = size(x_i,1);
            x_t_mk = x_t - k;

            q_sn = (x_i <= repmat(gumbel_quantile(x_i,a),x_t,1)) - (ones(x_t,n) .* a);

            d_sn = zeros(x_t_mk,n);
            d_sn(:,1) = q_sn(k+1:x_t,1);
            d_sn(:,2:n) = q_sn(1:x_t_mk,2:n);

            h_sn = d_sn.' * d_sn;

            if (det(h_sn) <= 1e-08)
                hi_sn = pinv(h_sn);
            else
                hi_sn = inv(h_sn);
            end

            cq_sn(i) = -hi_sn(1,2) / sqrt(hi_sn(1,1) * hi_sn(2,2));
        end 
    else
        for i = len:t
            x_i = x(1:i,:);
            x_t = size(x_i,1);
            x_t_mk = x_t - k;

            q_sn = (x_i <= repmat(gumbel_quantile(x_i,a),x_t,1)) - (ones(x_t,n) .* a);

            d_sn = zeros(x_t_mk,n);
            d_sn(:,1) = q_sn(k+1:x_t,1);
            d_sn(:,2:n) = q_sn(1:x_t_mk,2:n);

            h_sn = d_sn.' * d_sn;

            cq_sn(i) = h_sn(1,2) / sqrt(h_sn(1,1) * h_sn(2,2));
        end
    end

    cq = cq_sn(end);

    cqc = (cq_sn - repmat(cq,t,1)) .* (1:t).';
    cv0_bb = cqc(len:t);
    cv0_sn = (t * cq^2) / ((1 / t^2) * (cv0_bb.' * cv0_bb));
    cv0 = sqrt(civ * (cq^2 / cv0_sn));
    ci = [-cv0 cv0];

end

function q = gumbel_quantile(x,p)

    index = 1 + ((size(x,1) - 1) * p);
    low = floor(index);
    high = ceil(index);

    x = sort(x);
    x_low = x(low,:);
    x_high = x(high,:);

    h = max(index - low,0);
    q = (h .* x_high) + ((1 - h) .* x_low);

end

function [civ,cif] = validate_input_ci(cis,cif)

    persistent cif_allowed;
    persistent cis_allowed;
    persistent v;

    if (isempty(cif_allowed))
        cif_allowed = [0.00 0.01 0.03 0.05 0.10 0.15 0.20 0.30];
    end

    if (isempty(cis_allowed))
        cis_allowed = [0.005 0.010 0.025 0.050 0.100];
    end

    if (isempty(v))
        v = [
            129.15490  99.44085  66.00439 45.43917 28.06313;
            131.82880 101.21590  66.49058 45.48538 28.31850;
            131.83560 101.31700  67.55465 46.02739 28.51908;
            135.24000 103.32090  68.20319 46.48723 28.82658;
            139.34750 106.80970  71.07829 48.55132 30.01695;
            150.60740 115.90190  76.13708 51.38898 31.66900;
            166.53440 127.71000  83.45021 55.85094 34.03793;
            206.01210 155.00120 101.10960 67.53906 40.84930
        ];
    end

    [cis_ok,j] = ismember(cis,cis_allowed);

    if (~cis_ok)
        cis_allowed_text = [sprintf('%.3f',cis_allowed(1)) sprintf(', %.3f',cis_allowed(2:end))];
        error(['The value of ''cis'' is invalid. Expected input to have one of the following values: ' cis_allowed_text '.']);
    end

    [cif_ok,i] = ismember(cif,cif_allowed);

    if (~cif_ok)
        cif_allowed_text = [sprintf('%.2f',cif_allowed(1)) sprintf(', %.2f',cif_allowed(2:end))];
        error(['The value of ''cif'' is invalid. Expected input to have one of the following values: ' cif_allowed_text '.']);
    end

    civ = v(i,j);

end

function data = validate_input_data(data)

    n = size(data,2);

    if (n < 2)
        error('The value of ''data'' is invalid. Expected input to be a matrix with at least 2 columns.');
    end

end
