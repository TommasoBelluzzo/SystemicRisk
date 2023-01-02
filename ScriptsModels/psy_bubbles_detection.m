% [INPUT]
% p = A vector of floats [0,Inf) of length t representing the prices.
% cvm = A string representing the method for calculating the critical values (optional, default='WB'):
%   - 'FS' for finite sample critical values;
%   - 'WB' for wild bootstrap.
% cvq = A float [0.90,0.99] representing the quantile of the critical values (optional, default=0.90).
% lag_max = An integer [0,10] representing the maximum lag order to be evaluated for the Augmented Dickey-Fuller test (optional, default=0).
% lag_sel = A string representing the lag order selection criteria for the Augmented Dickey-Fuller test (optional, default='FIX'):
%   - 'AIC' for Akaike's Information Criterion;
%   - 'BIC' for Bayesian Information Criterion;
%   - 'FIX' to use a fixed lag order;
%   - 'FPE' for Final Prediction Error;
%   - 'HQIC' for Hannan-Quinn Information Criterion.
% mbd = An integer [3,Inf) representing the minimum duration of a bubble in days (optional, default=NaN).
%   If NaN is provided, then an optimal value based on the number of observations is used.
%
% [OUTPUT]
% bsadfs = A vector of floats (-Inf,Inf) and NaN values of length t representing the backward supremum Augmented Dickey-Fuller statistics.
% cvs = A vector of floats (-Inf,Inf) and NaN values of length t representing the critical values.
% detection = A boolean t-by-3 matrix where true values indicate the presence of:
%   - a bubble in the first column;
%   - a boom phase in the second column;
%   - a burst phase in the third column.
% breakdown = An integer k-by-6 matrix [1,Inf), where k is the number of detected bubbles, in which every column is a bubble property:
%   - the first column represents the starting day;
%   - the second column represents the peak day;
%   - the third column represents the ending day;
%   - the fourth column represents the overall duration;
%   - the fifth column represents the boom phase duration;
%   - the sixth column represents the burst phase duration.

function [bsadfs,cvs,detection,breakdown] = psy_bubbles_detection(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('p',@(x)validateattributes(x,{'double'},{'real' 'finite' 'nonnegative' 'vector' 'nonempty'}));
        ip.addOptional('cvm','WB',@(x)any(validatestring(x,{'FS' 'WB'})));
        ip.addOptional('cvq',0.95,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.90 '<=' 0.99 'scalar'}));
        ip.addOptional('lag_max',0,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 0 '<=' 10 'scalar'}));
        ip.addOptional('lag_sel','FIX',@(x)any(validatestring(x,{'AIC' 'BIC' 'FIX' 'FPE' 'HQIC'})));
        ip.addOptional('mbd',NaN,@(x)validateattributes(x,{'double'},{'real' 'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    [p,mbd] = validate_input(ipr.p,ipr.mbd);
    cvm = ipr.cvm;
    cvq = ipr.cvq;
    lag_max = ipr.lag_max;
    lag_sel = ipr.lag_sel;

    nargoutchk(4,4);

    [bsadfs,cvs,detection,breakdown] = psy_bubbles_detection_internal(p,cvm,cvq,lag_max,lag_sel,mbd);

end

function [bsadfs,cvs,detection,breakdown] = psy_bubbles_detection_internal(y,cvm,cvq,lag_max,lag_sel,mbd)

    t = numel(y);

    r0 = 0.01 + (1.8 / sqrt(t));
    w = floor(r0 * t);

    bsadfs = psy_test(y,w,lag_max,lag_sel,true);

    if (strcmp(cvm,'FS'))
        cvs = critical_values_fs(t,w,cvq,lag_max,lag_sel);
    else
        cvs = critical_values_wb(y,w,cvq,lag_max,lag_sel);
    end

    filler = NaN(w - 1,1);
    bsadfs = [filler; bsadfs];
    cvs = [filler; cvs];

    critical_points = bsadfs > cvs;
    indices = find(critical_points == 1);

    if (numel(indices) == 0)
        detection = false(t,3);
        breakdown = [];

        return;
    end

    diffs = [indices(2:end); indices(end)] - indices;
    trans = find(diffs ~= 1);
    
    bubbles_start = [indices(1); indices(trans(1:end-1) + 1)];
    bubbles_end = indices(trans);
    bubbles_duration = bubbles_end - bubbles_start + 1;

    if (isnan(mbd))
        mbd = floor(log(t));
    end

    mbm = mbd * 2;

    bubbles_filter = bubbles_duration < mbd;
    bubbles_start(bubbles_filter) = [];
    bubbles_end(bubbles_filter) = [];
    bubbles_duration(bubbles_filter) = [];

    k = numel(bubbles_duration);

    if (k == 0)
        detection = false(t,3);
        breakdown = [];

        return;
    end

    if (k >= 2)
        k = k - 1;

        while (k > 0)
            k1 = k + 1;
            bubble_next = bubbles_start(k1) - bubbles_end(k);
    
            if (bubble_next > mbm)
                k = k - 1;
                continue;
            end
    
            bubbles_end(k) = bubbles_end(k1);
            bubbles_duration(k) = bubbles_end(k) - bubbles_start(k) + 1;
    
            bubbles_start(k1) = [];
            bubbles_end(k1) = [];
            bubbles_duration(k1) = [];
    
            if (k == numel(bubbles_duration))
                k = k - 1;
            end
        end
    end

    k = numel(bubbles_duration);

    bubbles = false(t,1);
    booms = false(t,1);
    bursts = false(t,1);

    bubbles_peak = NaN(k,1);
    bubbles_duration_boom = NaN(k,1);
    bubbles_duration_burst = NaN(k,1);
    
    for i = 1:k
        bubble_start = bubbles_start(i);
        bubble_end = bubbles_end(i);
        bubble_duration = bubbles_duration(i);

        [~,peak] = max(y(bubble_start+1:bubble_end));
        bubbles_peak(i) = bubble_start + peak;
        bubbles_duration_boom(i) = peak;
        bubbles_duration_burst(i) = bubble_duration - peak;

        bubbles(bubble_start:bubble_end) = true;
    
        boom_start = bubble_start;
        boom_end = bubbles_peak(i);
        booms(boom_start:boom_end) = true;
    
        burst_start = boom_end + 1;
        burst_end = bubble_end;
        bursts(burst_start:burst_end) = true;
    end

    detection = [bubbles booms bursts];
    breakdown = [bubbles_start bubbles_peak bubbles_end bubbles_duration bubbles_duration_boom bubbles_duration_burst];

end

function [b,e,stat,lag] = adf(res,y,lag_max,lag_sel)

    t1 = numel(y) - 1;
    tlm = t1 - lag_max;

    lm1 = lag_max + 1;
    dof = tlm - 2;

    if (res)
        y0 = [];
    else
        y0 = y(1:t1);
    end

    x0 = [y0 ones(t1,1)];
    x0 = x0(lm1:t1,:);
    x0c = size(x0,2);

    dy = diff(y);
    dys = dy(lm1:t1);

    if (strcmp(lag_sel,'FIX'))
        if (lag_max > 0)
            x = [x0 zeros(tlm,lag_max)];

            for j = 1:lag_max
                x(:,x0c + j) = dy(lm1-j:t1-j);
            end
        else
            x = x0;
        end

        xt = x.';
        x2 = xt * x;

        b = x2 \ (xt * dys);
        e  = dys - (x * b);

        se = (e.' * e) ./ dof;
        sdi = sqrt(diag(se .* inv(x2)));
        stat = b(1) / sdi(1);

        lag = lag_max;
    else
        switch (lag_sel)
            case 'AIC'
                icf = @(npdf,k,d)((-2 * npdf) / k) + (2 * d);
            case 'BIC'
                icf = @(npdf,k,d)((-2 * npdf) / k) + (log(k) * d);
            case 'FPE'
                icf = @(npdf,k,d)((-2 * npdf) / k) + log((k + d + 1) / (k + d - 1));
            otherwise
                icf = @(npdf,k,d)((-2 * npdf) / k) + (2 * log(log(k)) * d);
        end

        ics = zeros(lm1,1);
        bs = cell(lm1,1);
        es = cell(lm1,1);
        stats = zeros(lm1,1);

        npdf_const = -0.5 * log(2 * pi());
        
        for k = 0:lag_max
            if (k > 0)
                x = [x0 zeros(tlm,k)];

                for j = 1:k
                    x(:,x0c + j) = dy(lm1-j:t1-j);
                end
            else
                x = x0;
            end

            xt = x.';
            x2 = xt * x;

            b = x2 \ (xt * dys);
            e  = dys - (x * b);

            se = (e.' * e) ./ dof;
            sdi = sqrt(diag(se .* inv(x2)));
            stat = b(1) / sdi(1);

            npdf = sum(npdf_const - (0.5 .* (e .^ 2)));

            k1 = k + 1;
            ics(k1) = icf(npdf,tlm,size(b,1) / tlm);
            bs{k1} = b;
            es{k1} = e;
            stats(k1) = stat;
        end
        
        [~,index] = min(ics);
        b = bs{index};
        e = es{index};
        stat = stats(index);
        lag = index - 1;
    end

end

function cvs = critical_values_fs(t,w,cvq,lag_max,lag_sel)

    m = 2000;
    y = cumsum(randn(t,m) + (1 / t));

    z = zeros(m,t - w + 1);

    parfor i = 1:m
        z(i,:) = psy_test(y(:,i),w,lag_max,lag_sel,false);
    end

    cvs = quantile(z,cvq).';

end

function cvs = critical_values_wb(y,w,cvq,lag_max,lag_sel)

    m = 500;
    t = numel(y);

    [b,e,~,lag] = adf(true,y,lag_max,lag_sel);

    dy = diff(y);
    dyb = zeros(w,m);
    dyb(1:lag,:) = repmat(dy(1:lag),1,m);
    
    ri = randi(numel(e),[(w + 1) m]);  
    rn = randn(w + 1,m);

    if (lag == 0)
        for j = 1:m      
            for i = lag+1:w
                v = i - lag;
                dyb(i,j) = rn(v,j) * e(ri(v,j));
            end
        end
    else
        for j = 1:m
            x = zeros(w,lag);

            for i = lag+1:w
                for k = 1:lag
                    x(i,k) = dyb(i - k,j); 
                end

                v = i - lag;
                dyb(i,j) = (x(i,:) * b(2:end)) + (rn(v,j) * e(ri(v,j)));
            end
        end
    end

    yb = cumsum([repmat(y(1),1,m); dyb]);

    z = zeros(m,2);

    parfor i = 1:m
        z(i,:) = psy_test(yb(:,i),w,lag_max,lag_sel,false);
    end

    cvs = ones(t - w + 1,1) .* quantile(max(z,[],2),cvq);

end

function bsadfs = psy_test(y,w,lag_max,lag_sel,parallel)

    t = length(y);
    y_slices = cell(t,1);

    for i = w:t
        y_slices{i} = y(1:i);
    end

    bsadfs = NaN(t,1);

    if (parallel)
        parfor i = w:t
            yi = y_slices{i};
            k = i - w + 1;
    
            v_bsadfs = zeros(k,1);
            
            for j = 1:k
                [~,~,stat,~] = adf(false,yi(j:end),lag_max,lag_sel);  
                v_bsadfs(j) = stat;
            end
            
            bsadfs(i) = max(v_bsadfs);
        end
    else
        for i = w:t
            yi = y_slices{i};
            k = i - w + 1;
    
            v_bsadfs = zeros(k,1);
            
            for j = 1:k
                [~,~,stat,~] = adf(false,yi(j:end),lag_max,lag_sel);  
                v_bsadfs(j) = stat;
            end
            
            bsadfs(i) = max(v_bsadfs);
        end
    end
    
    bsadfs = bsadfs(w:t);

end

function [p,mbd] = validate_input(p,mbd)

    p = p(:);
    t = numel(p);

    if (t < 5)
        error('The value of ''r'' is invalid. Expected input to be a vector containing at least 5 elements.');
    end

    if (~isnan(mbd))
        if (~isfinite(mbd))
            error('The value of ''mbd'' is invalid. Expected input to be finite.');
        end
    
        if (floor(mbd) ~= mbd)
            error('The value of ''mbd'' is invalid. Expected input to be an integer.');
        end

        b = ceil(0.2 * t);

        if ((mbd < 3) || (mbd > b))
            error(['The value of ''mbd'' is invalid. Expected input to have a value >= 5 and <= ' num2str(b) '.']);
        end
    end

end
