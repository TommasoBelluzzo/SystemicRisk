% [INPUT]
% r = A vector of floats (-Inf,Inf) of length t representing the logarithmic returns.
% v = A vector of floats [0,Inf) of length t representing the trading volumes.
% sv = A float t-by-k matrix (-Inf,Inf) representing the state variables.
% bw = An integer [21,252] representing the dimension of each rolling window.
% mem = A string representing the MEM type:
%   - 'B' for Baseline MEM;
%   - 'A' for Asymmetric MEM;
%   - 'P' for Asymmetric Power MEM;
%   - 'S' for Spline MEM.
% mag = An integer [1,Inf) obtained as 10^x representing the magnitude of logarithmic returns and trading volumes (optional, default=[]).
%
% [OUTPUT]
% illiq = A column vector of floats [0,Inf) of length t representing the ILLIQ indicator.
% illiqc = A column vector of floats [0,Inf) of length t representing the ILLIQ indicator with covariates if state variables are provided, an empty array otherwise.
% knots = A row vector of floats [0,Inf) containing the optimal number of knots if Spline MEM is used, an empty array otherwise.

function [illiq,illiqc,knots] = illiq_indicator(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('r',@(x)validateattributes(x,{'double'},{'real' 'finite' 'vector' 'nonempty'}));
        ip.addRequired('v',@(x)validateattributes(x,{'double'},{'real' 'finite' 'nonnegative' 'vector' 'nonempty'}));
        ip.addRequired('sv',@(x)validateattributes(x,{'double'},{'real' 'finite'}));
        ip.addRequired('bw',@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 21 '<=' 252 'scalar'}));
        ip.addRequired('mem',@(x)any(validatestring(x,{'A' 'B' 'P' 'S'})));
        ip.addOptional('mag',[],@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    [r,v,sv,mag] = validate_input(ipr.r,ipr.v,ipr.sv,ipr.mag);
    bw = ipr.bw;
    mem = ipr.mem;

    nargoutchk(2,3);

    [illiq,illiqc,knots] = illiq_indicator_internal(r,v,sv,bw,mem,mag);

end

function [illiq,illiqc,knots] = illiq_indicator_internal(r,v,sv,bw,mem,mag)

    alpha = 2 / (bw + 1);

    input = mag .* (abs(r) ./ v);
    input(~isfinite(input) | (input == 0)) = NaN;
    input(isnan(input)) = mean(input,'omitnan');

    if (any(strcmp(mem,{'A' 'P'})))
        input = [input r];
    end

    knots = [];

    [illiq,~,mem_params] = multiplicative_error(input,mem);
    illiq = [illiq(1); filter(alpha,[1 (alpha - 1)],illiq(2:end),(1 - alpha) * illiq(1))];
    illiq = (illiq - min(illiq)) ./ (max(illiq) - min(illiq));

    if (strcmp(mem,'S'))
        knots(1) = mem_params(1);
    end

    if (isempty(sv))
        illiqc = [];
    else
        [illiqc,~,mem_params] = multiplicative_error([input sv],mem);
        illiqc = [illiqc(1); filter(alpha,[1 (alpha - 1)],illiqc(2:end),(1 - alpha) * illiqc(1))];
        illiqc = (illiqc - min(illiqc)) ./ (max(illiqc) - min(illiqc));

        if (strcmp(mem,'S'))
            knots(2) = mem_params(1);
        end
    end

end

function [r,v,sv,mag] = validate_input(r,v,sv,mag)

    data = {r(:) v(:)};

    l = unique(cellfun(@numel,data));

    if (numel(l) ~= 1)
        error('The number of elements of ''r'' and ''v'' must be equal.');
    end

    if (l < 5)
        error('The value of ''r'' and ''v'' is invalid. Expected inputs to be vectors containing at least 5 elements.');
    end

    [r,v] = deal(data{:});

    if (~isempty(sv))
        if (size(sv,1) ~= l)
            error(['The value of ''sv'' is invalid. Expected input to be a matrix with ' num2str(l) ' rows.']);
        end
    end

    if (isempty(mag))
        mag_r = floor(round((log(abs(r)) ./ log(10)),15));
        mag_r(~isfinite(mag_r)) = [];
        mag_r = round(abs(mean(mag_r)),0);

        mag_v = floor(round((log(abs(v)) ./ log(10)),15));
        mag_v(~isfinite(mag_v)) = [];
        mag_v = round(mean(mag_v),0);

        mag = 10^(mag_r + mag_v);
    else
        if ((mag ~= 1) && (rem(mag,10) ~= 0))
            error('The value of ''mag'' is invalid. Expected input to be an integer obtained as 10^x.');
        end
    end

end
