% [INPUT]
% data = A numeric t-by-n matrix containing the time series.
% lags = An integer [1,3] representing the number of lags of the VAR model (optional, default=2).
% h = An integer [1,10] representing the prediction horizon (optional, default=4).
% fevd = A string (either 'generalized or 'orthogonal') representing the FEVD type (optional, default='generalized').
%
% [OUTPUT]
% vd = An numeric n-by-n matrix representing the variance decomposition of the network.

function vd = variance_decomposition(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data',@(x)validateattributes(x,{'numeric'},{'2d','nonempty','real','finite'}));
        ip.addOptional('lags',2,@(x)validateattributes(x,{'numeric'},{'scalar','integer','real','finite','>=',1,'<=',3}));
        ip.addOptional('h',4,@(x)validateattributes(x,{'numeric'},{'scalar','integer','real','finite','>=',1,'<=',10}));
        ip.addOptional('fevd','generalized',@(x)any(validatestring(x,{'generalized','orthogonal'})));
    end

    ip.parse(varargin{:});
    ipr = ip.Results;
    
    nargoutchk(1,1);
    
    vd = variance_decomposition_internal(ipr.data,ipr.lags,ipr.h,ipr.fevd);

end

function vd = variance_decomposition_internal(data,lags,h,fevd) 

    n = size(data,2);

    zeros_indices = find(~data);
    data(zeros_indices) = (((1e-10 - 1e-8) .* rand(numel(zeros_indices),1)) + 1e-8);

    if (verLessThan('MATLAB','9.4'))
        spec = vgxset('n',n,'nAR',lags,'Constant',true);
        model = vgxvarx(spec,data(lags+1:end,:),[],data(1:lags,:));
        
        vma = vgxma(model,h,1:h);
        vma.MA(2:h+1) = vma.MA(1:h);
        vma.MA{1} = eye(n);
        
        covariance = vma.Q;
    else
        spec = varm(n,lags);
        model = estimate(spec,data(lags+1:end,:),'Y0',data(1:lags,:));

        r = zeros(n * lags,n * lags);
        r(1:n,:) = cell2mat(model.AR);
        
        if (lags > 2)
            r(n+1:end,1:(end-n)) = eye((lags - 1) * n);
        end

        vma.MA{1,1} = eye(n);
        vma.MA{2,1} = r(1:n,1:n);

        if (h >= 3)
            for i = 3:h
                temp = r^i;
                vma.MA{i,1} = temp(1:n,1:n);
            end
        end
        
        covariance = model.Covariance;
    end
    
    irf = zeros(h,n,n);
    vds = zeros(h,n,n);
    
    if (strcmp(fevd,'generalized'))
        sigma = diag(covariance);

        for i = 1:n
            indices = zeros(n,1);
            indices(i,1) = 1;

            for j = 1:h
                irf(j,:,i) = (sigma(i,1) .^ -0.5) .* (vma.MA{j} * covariance * indices);
            end
        end
    else
        try
            p = chol(covariance,'lower');
        catch
            [v,d] = eig(covariance);  
            covariance = (v * max(d,0)) / v;
            p = chol(covariance,'lower');
        end

        for i = 1:n
            indices = zeros(n,1);
            indices(i,1) = 1;

            for j = 1:h
                irf(j,:,i) = vma.MA{j} * p * indices; 
            end
        end
    end

    irf_cs = cumsum(irf .^ 2);
    denominator = sum(irf_cs,3);

    for i = 1:n
        vds(:,:,i) = irf_cs(:,:,i) ./ denominator;     
    end

    vd = squeeze(vds(h,:,:));

end
