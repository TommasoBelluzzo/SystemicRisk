% [INPUT]
% ds = A structure representing the dataset.
% temp = A string representing the full path to the Excel spreadsheet used as a template for the results file.
% out = A string representing the full path to the Excel spreadsheet to which the results are written, eventually replacing the previous ones.
% bw = An integer [21,252] representing the dimension of each rolling window (optional, default=252).
% rr = A float [0,1] representing the recovery rate in case of default (optional, default=0.45).
% lst = A float or a vector of floats (0,Inf) representing the long-term to short-term liabilities ratio(s) used for the calculation of D2C and D2D default barriers (optional, default=3).
% car = A float [0.03,0.20] representing the capital adequacy ratio used to calculate the D2C (optional, default=0.08).
% c = An integer [50,1000] representing the number of simulated samples used to calculate the DIP (optional, default=100).
% l = A float [0.05,0.20] representing the importance sampling threshold used to calculate the DIP (optional, default=0.10).
% s = An integer [2,n], where n is the number of firms, representing the amount of systematic risk factors used to calculate the DIP (optional, default=2).
% op = A string (either 'BSM' for Black-Scholes-Merton or 'GC' for Gram-Charlier) representing the option pricing model used by the Systemic CCA framework (optional, default='BSM').
% k = A float [0.90,0.99] representing the confidence level used by the Systemic CCA framework (optional, default=0.95).
% analyze = A boolean that indicates whether to analyse the results and display plots (optional, default=false).
%
% [OUTPUT]
% result = A structure representing the original dataset inclusive of intermediate and final calculations.
% stopped = A boolean that indicates whether the process has been stopped through user input.

function [result,stopped] = run_default(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('ds',@(x)validateattributes(x,{'struct'},{'nonempty'}));
        ip.addRequired('temp',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addRequired('out',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addOptional('bw',252,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 21 '<=' 252 'scalar'}));
        ip.addOptional('rr',0.45,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0 '<=' 1 'scalar'}));
        ip.addOptional('lst',3,@(x)validateattributes(x,{'double'},{'real' 'finite' '>' 0 'vector'}));
        ip.addOptional('car',0.08,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.03 '<=' 0.20 'scalar'}));
        ip.addOptional('c',100,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 50 '<=' 1000 'scalar'}));
        ip.addOptional('l',0.10,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.05 '<=' 0.20 'scalar'}));
        ip.addOptional('s',2,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 2 'scalar'}));
        ip.addOptional('op','BSM',@(x)any(validatestring(x,{'BSM' 'GC'})));
        ip.addOptional('k',0.95,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.90 '<=' 0.99 'scalar'}));
        ip.addOptional('analyze',false,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    ds = validate_dataset(ipr.ds,'default');
    temp = validate_template(ipr.temp);
    out = validate_output(ipr.out);
    s = validate_s(ipr.s,ds.N);
    bw = ipr.bw;
    rr = ipr.rr;
    lst = validate_lst(ipr.lst,ds.N);
    car = ipr.car;
    c = ipr.c;
    l = ipr.l;
    op = ipr.op;
    k = ipr.k;
    analyze = ipr.analyze;
    
    nargoutchk(1,2);
    
    [result,stopped] = run_default_internal(ds,temp,out,bw,rr,lst,car,c,l,s,op,k,analyze);

end

function [result,stopped] = run_default_internal(ds,temp,out,bw,rr,lst,car,c,l,s,op,k,analyze)

    result = [];
    stopped = false;
    e = [];

    ds = initialize(ds,bw,rr,lst,car,c,l,s,op,k);
    n = ds.N;
    t = ds.T;
    
    step_1 = 0.1;
    step_2 = 1 - step_1;

    rng(double(bitxor(uint16('T'),uint16('B'))));
    cleanup_1 = onCleanup(@()rng('default'));

    bar = waitbar(0,'Initializing default measures...','CreateCancelBtn',@(src,event)setappdata(gcbf(),'Stop',true));
    setappdata(bar,'Stop',false);
    cleanup_2 = onCleanup(@()delete(bar));
    
    pause(1);
    waitbar(0,bar,'Calculating default measures (step 1 of 2)...');
    pause(1);

    try

        r = max(0,ds.RiskFreeRate);

        firms_data = extract_firms_data(ds,{'Capitalizations' 'CDS' 'Liabilities'});
        
        futures(1:n) = parallel.FevalFuture;
        futures_max = 0;
        futures_results = cell(n,1);

        for i = 1:n
            offsets = [ds.Defaults(i) ds.Insolvencies(i)] - 1;
            futures(i) = parfeval(@main_loop_1,1,firms_data{i},offsets,r,ds.ST(i),ds.DT(i),ds.LCAR,ds.OP);
        end

        for i = 1:n
            if (getappdata(bar,'Stop'))
                stopped = true;
                break;
            end
            
            [future_index,value] = fetchNext(futures);
            futures_results{future_index} = value;
            
            futures_max = max([future_index futures_max]);
            waitbar(step_1 * ((futures_max - 1) / n),bar);

            if (getappdata(bar,'Stop'))
                stopped = true;
                break;
            end
        end

    catch e
    end
    
    try
        cancel(futures);
    catch
    end
    
    if (~isempty(e))
        delete(bar);
        rethrow(e);
    end
    
    if (stopped)
        delete(bar);
        return;
    end
    
    pause(1);
    waitbar(step_1,bar,'Finalizing default measures (step 1 of 2)...');
    pause(1);

    try
        ds = finalize_1(ds,futures_results);
    catch e
        delete(bar);
        rethrow(e);
    end
    
    pause(1);
    waitbar(step_1,bar,'Calculating default measures (step 2 of 2)...');
    pause(1);
    
    try

        r = distress_data(ds.Returns,ds.Insolvencies);
        windows_r = extract_rolling_windows(r,ds.BW);

        cds = distress_data(ds.CDS,ds.Insolvencies);
        lb = distress_data(ds.Liabilities,ds.Insolvencies);

        cl = ds.SCCAContingentLiabilities;
        cl(isnan(cl)) = 0;
        windows_cl = extract_rolling_windows(cl,ds.BW);

        futures(1:t) = parallel.FevalFuture;
        futures_max = 0;
        futures_results = cell(t,1);

        for i = 1:t
            futures(i) = parfeval(@main_loop_2,1,windows_r{i},cds(i,:),lb(i,:),ds.LGD,ds.C,ds.L,ds.S,windows_cl{i},ds.Q,ds.QDiff);
        end

        for i = 1:t
            if (getappdata(bar,'Stop'))
                stopped = true;
                break;
            end
            
            [future_index,value] = fetchNext(futures);
            futures_results{future_index} = value;
            
            futures_max = max([future_index futures_max]);
            waitbar(step_1 + (step_2 * ((futures_max - 1) / t)),bar);

            if (getappdata(bar,'Stop'))
                stopped = true;
                break;
            end
        end

    catch e
    end
    
    try
        cancel(futures);
    catch
    end

    if (~isempty(e))
        delete(bar);
        rethrow(e);
    end
    
    if (stopped)
        delete(bar);
        return;
    end
    
    pause(1);
    waitbar(1,bar,'Finalizing default measures (step 2 of 2)...');
    pause(1);

    try
        ds = finalize_2(ds,futures_results);
    catch e
        delete(bar);
        rethrow(e);
    end

    pause(1);
    waitbar(1,bar,'Writing default measures...');
    pause(1);
    
    try
        write_results(ds,temp,out);
        delete(bar);
    catch e
        delete(bar);
        rethrow(e);
    end
    
    if (analyze)
        safe_plot(@(id)plot_distances(ds,id));
        safe_plot(@(id)plot_sequence(ds,'D2D',true,id));
        safe_plot(@(id)plot_sequence(ds,'D2C',true,id));
        safe_plot(@(id)plot_dip(ds,id));
        safe_plot(@(id)plot_scca(ds,id));
        safe_plot(@(id)plot_sequence(ds,'SCCA Expected Losses',false,id));
        safe_plot(@(id)plot_sequence(ds,'SCCA Contingent Liabilities',false,id));
    end
    
    result = ds;

end

%% DATA

function ds = initialize(ds,bw,rr,lst,car,c,l,s,op,k)

    n = ds.N;
    t = ds.T;

    q = [(0.900:0.025:0.975) 0.99];

    ds.A = 1 - k;
    ds.BW = bw;
    ds.C = c;
    ds.CAR = car;
    ds.DT = max(0.5,0.7 - (0.3 .* (1 ./ lst)));
    ds.K = k;
    ds.L = l;
    ds.LCAR = 1 / (1 - car);
    ds.LGD = 1 - rr;
    ds.LST = lst;
    ds.OP = op;
    ds.Q = q(q >= k);
    ds.QDiff = diff([ds.Q 1]);
    ds.RR = rr;
    ds.S = s;
    ds.ST =  1 ./ (1 + lst);

    car_label = sprintf('%.0f%%',(ds.CAR * 100));
    ds.LabelsIndicators = {'Average D2D' 'Average D2C' 'Portfolio D2D' 'Portfolio D2C' 'DIP' 'SCCA Joint ES'};
    ds.LabelsSheet = {'D2D' ['D2C (CAR=' car_label ')'] 'SCCA Expected Losses' 'SCCA Contingent Liabilities' 'Indicators'};
    ds.LabelsSheetSimple = {'D2D' 'D2C' 'SCCA Expected Losses' 'SCCA Contingent Liabilities' 'Indicators'};

    ds.D2D = NaN(t,n);
    ds.D2C = NaN(t,n);

    ds.SCCAAlphas = NaN(t,n);
    ds.SCCAExpectedLosses = NaN(t,n);
    ds.SCCAContingentLiabilities = NaN(t,n);
    ds.SCCAJointVaRs = NaN(t,numel(ds.Q));

    ds.Indicators = NaN(t,numel(ds.LabelsIndicators));

end

function ds = finalize_1(ds,window_results)
  
    n = ds.N;

    for i = 1:n
        window_result = window_results{i};

        ds.D2D(1:window_result.Offset1,i) = window_result.D2D;
        ds.D2C(1:window_result.Offset1,i) = window_result.D2C;

        ds.SCCAAlphas(1:window_result.Offset2,i) = window_result.SCCAAlphas;
        ds.SCCAExpectedLosses(1:window_result.Offset2,i) = window_result.SCCAExpectedLosses;
        ds.SCCAContingentLiabilities(1:window_result.Offset2,i) = window_result.SCCAContingentLiabilities;
    end

    [d2d_avg,d2c_avg,d2d_por,d2c_por] = calculate_overall_distances(ds);
    ds.Indicators(:,1) = d2d_avg;
    ds.Indicators(:,2) = d2c_avg;
    ds.Indicators(:,3) = d2d_por;
    ds.Indicators(:,4) = d2c_por;

end

function ds = finalize_2(ds,window_results)

    t = ds.T;

    for i = 1:t
        window_result = window_results{i};
        
        ds.SCCAJointVaRs(i,:) = window_result.SCCAJointVaRs;

        ds.Indicators(i,5) = window_result.DIP;
        ds.Indicators(i,6) = window_result.SCCAJointES;
    end
    
    w = round(nthroot(ds.BW,1.81),0); 
    ds.Indicators(:,5) = sanitize_data(ds.Indicators(:,5),ds.DatesNum,w,[]);

end

function lst = validate_lst(lst,n)

    if (isscalar(lst))
        lst = ones(1,n) .* lst;
    else
        if (numel(lst) ~= n)
            error(['The number of lst coefficients, when specified as a vector, must be equal to the number of firms (' num2str(n) ').']);
        end
        
        lst = lst(:).';
    end

end

function out_file = validate_output(out_file)

    [path,name,extension] = fileparts(out_file);

    if (~strcmp(extension,'.xlsx'))
        out_file = fullfile(path,[name extension '.xlsx']);
    end
    
end

function s = validate_s(s,n)

    if (s > n)
        error(['The amount of systematic risk factors used to calculate the DIP must be less than or equal to the number of firms (' num2str(n) ').']);
    end
    
end

function out_temp = validate_template(out_temp)

    if (exist(out_temp,'file') == 0)
        error('The template file could not be found.');
    end
    
    if (ispc())
        [file_status,file_sheets,file_format] = xlsfinfo(out_temp);
        
        if (isempty(file_status) || ~strcmp(file_format,'xlOpenXMLWorkbook'))
            error('The dataset file is not a valid Excel spreadsheet.');
        end
    else
        [file_status,file_sheets] = xlsfinfo(out_temp);
        
        if (isempty(file_status))
            error('The dataset file is not a valid Excel spreadsheet.');
        end
    end
    
    sheets = {'D2D' 'D2C' 'SCCA Expected Losses' 'SCCA Contingent Liabilities' 'Indicators'};

    if (~all(ismember(sheets,file_sheets)))
        error(['The template must contain the following sheets: ' sheets{1} sprintf(', %s',sheets{2:end}) '.']);
    end
    
    if (ispc())
        try
            excel = actxserver('Excel.Application');
            excel_wb = excel.Workbooks.Open(res,0,false);

            for i = 1:numel(sheets)
                excel_wb.Sheets.Item(sheets{i}).Cells.Clear();
            end
            
            excel_wb.Save();
            excel_wb.Close();
            excel.Quit();

            delete(excel);
        catch
        end
    end

end

function write_results(ds,temp,out)

    [out_path,~,~] = fileparts(out);

    try
        if (exist(out_path,'dir') ~= 7)
            mkdir(out_path);
        end

        if (exist(out,'file') == 2)
            delete(out);
        end
    catch
        error('A system I/O error occurred while writing the results.');
    end
    
    copy_result = copyfile(temp,out,'f');
    
    if (copy_result == 0)
        error('The output file could not be created from the template file.');
    end

    dates_str = cell2table(ds.DatesStr,'VariableNames',{'Date'});

    for i = 1:(numel(ds.LabelsSheetSimple) - 1)
        sheet = ds.LabelsSheetSimple{i};
        measure = strrep(sheet,' ','');

        tab = [dates_str array2table(ds.(measure),'VariableNames',ds.FirmNames)];
        writetable(tab,out,'FileType','spreadsheet','Sheet',sheet,'WriteRowNames',true);
    end

    tab = [dates_str array2table(ds.Indicators,'VariableNames',strrep(ds.LabelsIndicators,' ','_'))];
    writetable(tab,out,'FileType','spreadsheet','Sheet','Indicators','WriteRowNames',true);    

    if (ispc())
        try
            excel = actxserver('Excel.Application');
        catch
            return;
        end

        try
            exc_wb = excel.Workbooks.Open(out,0,false);

            for i = 1:numel(ds.LabelsSheet)
                exc_wb.Sheets.Item(ds.LabelsSheetSimple{i}).Name = ds.LabelsSheet{i};
            end
            
            exc_wb.Save();
            exc_wb.Close();
            excel.Quit();
        catch
        end
        
        try
            delete(excel);
        catch
        end
    end

end

%% MEASURES

function window_results = main_loop_1(firm_data,offsets,r,st,dt,lcar,op)

    t = size(firm_data,1);

    offset1 = min(offsets(1),t);
    cp_o = max(1e-6,firm_data(1:offset1,1));
    lb_o = max(1e-6,firm_data(1:offset1,3));
    db_o = (lb_o .* st) + (dt .* (lb_o .* (1 - st)));
    r_o = r(1:offset1);

    [va,va_m] = kmv_model(cp_o,db_o,r_o,1,op);
    [d2d,d2c] = calculate_distances(va,va_m,db_o,r_o,1,lcar);
    
    offset2 = min(min(offsets),t);
    cp_o = max(1e-6,firm_data(1:offset2,1));
    lb_o = max(1e-6,firm_data(1:offset2,3));
    db_o = (lb_o .* st) + (dt .* (lb_o .* (1 - st)));
    r_o = r(1:offset2);
    cds_o = firm_data(1:offset2,2);

    [va,va_m] = kmv_model(cp_o,db_o,r_o,1,op);
    [el,cl,a] = calculate_scca_values(va,va_m,db_o,r_o,cds_o,1,op);

    window_results = struct();
    window_results.Offset1 = offset1;
    window_results.Offset2 = offset2;
    window_results.D2D = d2d;
    window_results.D2C = d2c;
    window_results.SCCAAlphas = a;
    window_results.SCCAContingentLiabilities = cl;
    window_results.SCCAExpectedLosses = el;

end

function window_results = main_loop_2(r,cds,lb,lgd,c,l,s,window_cl,q,q_diff)

    window_results = struct();

    dip = calculate_dip(r,cds,lb,lgd,c,l,s);
    window_results.DIP = dip;

    [scca_joint_vars,scca_joint_es] = calculate_scca_indicators(window_cl,q,q_diff);
    window_results.SCCAJointVaRs = scca_joint_vars;
    window_results.SCCAJointES = scca_joint_es;

end

function dip = calculate_dip(r,cds,lb,lgd,c,l,s)

    indices = sum(isnan(r),1) == 0;
    n = sum(indices);

    [dt,dw,ead,ead_volume,lgd] = estimate_default_parameters(cds(indices),lb(indices),n,lgd);
    b = estimate_factor_loadings(r(:,indices),s);

    bi = floor(c * 0.2);
    c2 = c^2;

    a = zeros(5,1);
    
    for iter = 1:5 
        mcmc_p = slicesample(rand(1,s),c,'PDF',@(x)zpdf(x,dt,ead,lgd,b,l),'Thin',3,'BurnIn',bi);
        [mu,sigma,weights] = gmm_fit(mcmc_p,2);  
        [z,g] = gmm_evaluate(mu,sigma,weights,c);

        phi = normcdf((repmat(dt.',c,1) - (z * b.')) ./ (1 - repmat(sum(b.^2,2).',c,1)).^0.5);
        [theta,theta_p] = exponential_twist(phi,dw,l);

        losses = sum(repelem(dw.',c2,1) .* ((repelem(theta_p,c,1) >= rand(c2,n)) == 1),2);
        psi = sum(log((phi .* exp(repmat(theta,1,n) .* repmat(dw.',c,1))) + (1 - phi)),2);

        lr_z = repelem(mvnpdf(z) ./ g,c,1);
        lr_e = exp(-(repelem(theta,c,1) .* losses) + repelem(psi,c,1));
        lr = lr_z .* lr_e;

        a(iter) = mean((losses > l) .* lr);
    end
    
    dip = mean(a) * ead_volume;

end

function [d2d,d2c] = calculate_distances(va,va_m,db,r,t,lcar)

    s = va_m(1);
    rst = (r + (0.5 * s^2)) * t;
    st = s * sqrt(t);

    d1 = (log(va ./ db) + rst) ./ st;
    d2d = d1 - st;

    d1 = (log(va ./ (lcar .* db)) + rst) ./ st;
    d2c = d1 - st;

end

function [d2d_avg,d2c_avg,d2d_por,d2c_por] = calculate_overall_distances(ds)

    n = ds.N;
    t = ds.T;

    cp = ds.Capitalizations;
    lb = ds.Liabilities;
    r = max(0,ds.RiskFreeRate);

    weights = cp ./ repmat(sum(cp,2,'omitnan'),1,n);
    d2d_avg = sum(ds.D2D .* weights,2,'omitnan');
    d2c_avg = sum(ds.D2C .* weights,2,'omitnan');

    cp = max(1e-6,sum(cp,2,'omitnan'));
    lbs = lb .* repmat(ds.ST,t,1);
    lbl = repmat(ds.DT,t,1) .* (lb .* (1 - repmat(ds.ST,t,1)));
    db = max(1e-6,sum(lbs + lbl,2,'omitnan'));

    [va,va_m] = kmv_model(cp,db,r,1,ds.OP);
    [d2d_por,d2c_por] = calculate_distances(va,va_m,db,r,1,ds.LCAR);
    
end

function [joint_vars,joint_es] = calculate_scca_indicators(data,q,q_diff)

    persistent options;

    if (isempty(options))
        options = optimset(optimset(@fmincon),'Algorithm','sqp','Diagnostics','off','Display','off','LargeScale','off');
    end

    [t,n] = size(data);
    data_sorted = sort(data,1);
    
    xi_s = (1:floor(t / 4)).';
    xi_a = sqrt(log((t - xi_s) ./ t) ./ log(xi_s ./ t));
    xi_q0 = xi_s;
    xi_q1 = floor(t .* (xi_s ./ t).^xi_a);
    xi_q2 = t - xi_s;
    xi_r = (data_sorted(xi_q2,:) - data_sorted(xi_q1,:)) ./ max(1e-8,(data_sorted(xi_q1,:) - data_sorted(xi_q0,:)));
        
    xi = sum([zeros(1,n); -(log(xi_r) ./ (ones(1,n) .* log(xi_a)))]).' ./ xi_s(end);
    xi_positive = xi > 0;
    xi(xi_positive) = max(0.01,min(2,xi(xi_positive)));
    xi(~xi_positive) = max(-1,min(-0.01,xi(~xi_positive)));

    ms_d = floor(t / 10);
    ms_s = ((ms_d+1):(t-ms_d)).';
    ms_q = -log((1:t).' ./ (t + 1));
    
    mu = zeros(n,1);
    sigma = zeros(n,1);
    
    for j = 1:n
        y = (ms_q.^-xi(j) - 1) ./ xi(j);
        b = regress(data_sorted(ms_s,j),[ones(numel(ms_s),1) y(ms_s)]);
        
        mu(j) = b(1);
        sigma(j) = b(2);
    end
    
    d_p = tiedrank(data) ./ (t + 1);
    d_y = -log(d_p);
    d_v = (d_y ./ repmat(mean(d_y,1),t,1)) ./ (ones(size(data)) .* (1 / n));
    d = min(1,max(1 / mean(min(d_v,[],2)),1 / n));

    x0_mu = n * mean(mu);
    x0_sigma = sqrt(n) * mean(sigma);
    x0_xi = mean(xi);
    
    joint_vars = zeros(1,numel(q));

    for j = 1:numel(q)
        lhs = -log(q(j)) / d;

        x0 = (x0_mu + (x0_sigma / x0_xi) * (lhs^-x0_xi - 1));
        v0 = (1 + (x0_xi .* ((x0 - x0_mu) ./ x0_sigma))) .^ -(1 ./ x0_xi);

        e = [];

        try
            [joint_var,~,ef] = fmincon(@(x)objective(x,v0,lhs,n,mu,sigma,xi),x0,[],[],[],[],0,Inf,[],options);
        catch e
        end

        if (~isempty(e) || (ef <= 0))
            joint_vars(j) = 0;
        else
            joint_vars(j) = joint_var;
        end
    end

    indices = joint_vars > 0;

    if (any(indices))
        joint_es = sum(joint_vars(indices) .* q_diff(indices)) / sum(q_diff(indices));
    else
        joint_es = 0;
    end

    function y = objective(x,v,lhs,n,mu,sigma,xi)

        um = repelem(v,n,1);

        um_check = (xi .* (repelem(x,20,1) - mu)) ./ sigma;
        um_valid = isfinite(um_check) & (um_check > -1);
        
        x = repelem(x,sum(um_valid),1);
        mu = mu(um_valid);
        sigma = sigma(um_valid);
        xi = xi(um_valid);
        um(um_valid) = (1 + (xi .* ((x - mu) ./ sigma))) .^ -(1 ./ xi);

        y = (sum(um) - lhs)^2;

    end

end

function [el,cl,a] = calculate_scca_values(va,va_m,db,r,cds,t,op)

    s = va_m(1);
    st = s * sqrt(t);

    dbd = db .* exp(-r.* t);

    d1 = (log(va ./ db) + ((r + (0.5 * s^2)) .* t)) ./ st;
    d2 = d1 - st;

    put_price = (dbd .* normcdf(-d2)) - (va .* normcdf(-d1));
    
    if (strcmp(op,'GC'))
        g = va_m(2);
        k = va_m(3);

        t1 = (g / 6) .* ((2 * s) - d1);
        t2 = (k / 24) .* (1 - d1.^2 + (3 .* d1 .* s) - (3 * s^2));
        
        put_price = put_price - (va .* normcdf(d1) .* s .* (t1 - t2));
    end

    put_price = max(0,put_price);

    rd = dbd - put_price;

    cds_put_price = dbd .* (1 - exp(-cds .* max(0.5,((db ./ rd) - 1)) .* t));
    cds_put_price = min(cds_put_price,put_price);  
    
    a = max(0,min(1 - (cds_put_price ./ put_price),1));
    a(~isreal(a)) = 0;
    
    el = put_price;
    cl = el .* a;

end

function b = estimate_factor_loadings(r,f)

    rho = corr(r);
    f0 = eye(size(rho,1)) * 0.2;

    count = 0;
    error = 0.8;

    while ((count < 100) && (error > 0.01))
        [v,d] = eig(rho - f0,'vector');

        [~,sort_indices] = sort(d,'descend');
        sort_indices = sort_indices(1:f);

        d = diag(d(sort_indices));
        v = v(:,sort_indices);
        b = v * sqrt(d);

        f1 = diag(1 - diag(b * b.'));
        delta = f1 - f0;

        f0 = f1;

        count = count + 1;
        error = trace(delta * delta.');
    end

end

function [dt,dw,ead,ead_volume,lgd] = estimate_default_parameters(cds,liabilities,n,lgd)

    dt = norminv(1 - exp(-cds ./ lgd)).';

    liabilities_sum = sum(liabilities);
    ead = (liabilities / sum(liabilities_sum)).';
    ead_volume = liabilities_sum;

    if (lgd > 0.5)
        lgd = mean(cumsum(randtri((2 * lgd) - 1,lgd,1,[n 1000])),2);
    else
        lgd = mean(cumsum(randtri(0,lgd,1,[n 1000])),2);
    end

    dw = ead .* lgd;

end

function [theta,theta_p] = exponential_twist(phi,dw,l)

    persistent options;

    if (isempty(options))
        options = optimset(optimset(@fminunc),'Diagnostics','off','Display','off','LargeScale','off');
    end
    
    [c,n] = size(phi);

    theta = zeros(c,1);
    theta_p = phi;

    dw = [dw zeros(n,1)];

    for i = 1:c
        phi_i = phi(i,:).';
        p = [phi_i (1 - phi_i)];

        threshold = sum(sum(dw .* p,2),1);

        if (l > threshold)
            if (i == 1)
                x0 = 0;
            else
                x0 = theta(i-1);
            end

            e = [];

            try
                [t,~,ef] = fminunc(@(x)objective(x,p,w,l),x0,options);
            catch e
            end

            if (isempty(e) && (ef > 0))
                theta(i) = t;

                twist = p .* exp(dw .* t(end));
                theta_p(i,:) = twist(:,1) ./ sum(twist,2);
            end
        end
    end
    
    function y = objective(x,p,w,l)

        y = sum(log(sum(p .* exp(w .* x),2)),1) - (x * l);

    end

end

function [z,g] = gmm_evaluate(mu,sigma,weights,c)

    indices = datasample(1:numel(weights),c,'Replace',true,'Weights',weights);
    z = mvnrnd(mu(indices,:),sigma(:,:,indices),c);

    g = zeros(c,1);

    for i = 1:c
        g(i) = sum(mvnpdf(z(i,:),mu,sigma) .* weights);
    end

end

function [mu,sigma,weights] = gmm_fit(x,gm)

    [c,s] = size(x);

    m = x(randsample(c,gm),:);
    [~,indices] = max((x * m.') - repmat(dot(m,m,2).' / 2,c,1),[],2);
    [u,~,indices] = unique(indices);

    while (numel(u) ~= gm)
        m = x(randsample(c,gm),:);
        [~,indices] = max((x * m.') - repmat(dot(m,m,2).' / 2,c,1),[],2);
        [u,~,indices] = unique(indices);
    end
    
    r = zeros(c,gm);
    r(sub2ind([c gm],1:c,indices.')) = 1;

    [~,indices] = max(r,[],2);
    r = r(:,unique(indices));

    llh_old = -Inf;
    count = 1;
    converged = false;

    while ((count < 10000) && ~converged)
        count = count + 1;

        rk = size(r,2);
        rs = sum(r,1).';
        rq = sqrt(r);

        mu = (r.' * x) .* repmat(1 ./ rs,1,s);
        sigma = zeros(s,s,rk);
        rho = zeros(c,rk);
        weights = rs ./ c;

        for j = 1:rk
            x0 = x - repmat(mu(j,:),c,1);

            o = x0 .* repmat(rq(:,j),1,s);
            h = ((o.' * o) ./ rs(j)) + (eye(s) .* 1e-6);
            sigma(:,:,j) = h;

            v = chol(h,'upper');
            q0 = v.' \ x0.';
            q1 = dot(q0,q0,1);
            nc = (s * log(2 * pi())) + (2 * sum(log(diag(v))));
            rho(:,j) = (-(nc + q1) / 2) + log(weights(j));
        end

        rho_max = max(rho,[],2);
        t = rho_max + log(sum(exp(rho - repmat(rho_max,1,rk)),2));
        fi = ~isfinite(rho_max);
        t(fi) = rho_max(fi);
        llh = sum(t) / c;

        r = exp(rho - repmat(t,1,rk));

        [~,indices] = max(r,[],2);
        u = unique(indices);

        if (size(r,2) ~= numel(u))
            r = r(:,u);
        else
            converged = (llh - llh_old) < (1e-8 * abs(llh));
        end

        llh_old = llh;
    end

end

function [va,va_m] = kmv_model(eq,db,r,t,op)

    df = exp(-r.* t);

    k = numel(r);
    sk = sqrt(k);

    va = eq + (db .* df);
    va_r = diff(log(va));
    va_s = sqrt(252) * std(va_r);

    sst = va_s * sqrt(t);
    d1 = (log(va ./ db) + ((r + (0.5 * va_s^2)) .* t)) ./ sst;
    d2 = d1 - sst;
    n1 = normcdf(d1);
    n2 = normcdf(d2);

    va_old = va;
    va = eq + ((va .* (1 - n1)) + (db .* df .* n2));
    
    count = 0;
    error = norm(va - va_old) / sk;

    while ((count < 10000) && (error > 1e-8))
        sst = va_s * sqrt(t);
        d1 = (log(va ./ db) + ((r + (0.5 * va_s^2)) .* t)) ./ sst;
        d2 = d1 - sst;
        n1 = normcdf(d1);
        n2 = normcdf(d2);

        va_old = va;
        va = eq + ((va .* (1 - n1)) + (db .* df .* n2));
        va_r = diff(log(va));
        va_s = sqrt(252) * std(va_r);

        count = count + 1;
        error = norm(va - va_old) / sk;
    end
    
    if (strcmp(op,'BSM'))
        va_m = [va_s NaN(1,2)];
    else
        va_g = skewness(va_r,0) / sqrt(252);
        va_k = (kurtosis(va_r,0) - 3) / 252;
        
        va_m = [va_s va_g va_k];
    end

end

function r = randtri(a,b,c,size)

    d = (b - a) / (c - a);

    p = rand(size);
    r = p;

    t = ((p >= 0) & (p <= d));
    r(t) = a + sqrt(p(t) * (b - a) * (c - a));

    t = ((p <= 1) & (p > d));
    r(t) = c - sqrt((1 - p(t)) * (c - b) * (c - a));

end

function p = zpdf(z,dt,ead,lgd,b,l) 

    p0 = normcdf((dt - (b * z.')) ./ (1 - sum(b.^2,2)).^0.5);
    mu = sum(ead .* lgd .* p0);
    sigma = sqrt((ead.' .^ 2) * sum((-1 .* lgd).^2 .* p0 .* (1 - p0),2));

    p = max(1e-16,(1 - normcdf((l - mu) / sigma)) * mvnpdf(z));

end

%% PLOTTING

function plot_distances(ds,id)

    distances = ds.Indicators(:,1:4);

    y_min = min(min(min(distances)),-1);
    y_max = max(max(distances));
    y_limits = plot_limits(distances,0.1,[],[],-1);
    
    y_ticks = floor(y_min):0.5:ceil(y_max);
    y_ticks_labels = arrayfun(@(x)sprintf('%.1f',x),y_ticks,'UniformOutput',false);

    f = figure('Name','Default Measures > Distances','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    sub_1 = subplot(2,2,1);
    plot(sub_1,ds.DatesNum,smooth_data(distances(:,1)),'Color',[0.000 0.447 0.741]);
    hold on;
        p = plot(sub_1,ds.DatesNum,zeros(ds.T,1),'Color',[1 0.4 0.4]);
    hold off;
    title(sub_1,ds.LabelsIndicators{1});
    
    sub_2 = subplot(2,2,2);
    plot(sub_2,ds.DatesNum,smooth_data(distances(:,3)),'Color',[0.000 0.447 0.741]);
    hold on;
        plot(sub_2,ds.DatesNum,zeros(ds.T,1),'Color',[1 0.4 0.4]);
    hold off;
    title(sub_2,ds.LabelsIndicators{2});
    
    sub_3 = subplot(2,2,3);
    plot(sub_3,ds.DatesNum,smooth_data(distances(:,2)),'Color',[0.000 0.447 0.741]);
    hold on;
        plot(sub_3,ds.DatesNum,zeros(ds.T,1),'Color',[1 0.4 0.4]);
    hold off;
    title(sub_3,ds.LabelsIndicators{3});
    
    sub_4 = subplot(2,2,4);
    plot(sub_4,ds.DatesNum,smooth_data(distances(:,4)),'Color',[0.000 0.447 0.741]);
    hold on;
        plot(sub_4,ds.DatesNum,zeros(ds.T,1),'Color',[1 0.4 0.4]);
    hold off;
    title(sub_4,ds.LabelsIndicators{4});
    
    set([sub_1 sub_2 sub_3 sub_4],'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
    set([sub_1 sub_2 sub_3 sub_4],'YLim',y_limits,'YTick',y_ticks,'YTickLabel',y_ticks_labels);
    set([sub_1 sub_2 sub_3 sub_4],'XGrid','on','YGrid','on');
    
    if (ds.MonthlyTicks)
        date_ticks([sub_1 sub_2 sub_3 sub_4],'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        date_ticks([sub_1 sub_2 sub_3 sub_4],'x','yyyy','KeepLimits');
    end

    l = legend(sub_1,p,'Default Threshold','Location','best');
    set(l,'Units','normalized');
    l_position = get(l,'Position');
    set(l,'Position',[0.4683 0.4799 l_position(3) l_position(4)]);

    figure_title('Distances');

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_dip(ds,id)

    dip = ds.Indicators(:,5);
    y = smooth_data(dip);

    f = figure('Name','Default Measures > Distress Insurance Premium','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    sub_1 = subplot(1,6,1:5);
    plot(sub_1,ds.DatesNum,y,'Color',[0.000 0.447 0.741]);
    set(sub_1,'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
    set(sub_1,'XGrid','on','YGrid','on');
    
    if (ds.MonthlyTicks)
        date_ticks(sub_1,'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        date_ticks(sub_1,'x','yyyy','KeepLimits');
    end
    
    sub_2 = subplot(1,6,6);
    boxplot(sub_2,y,'Notch','on','Symbol','k.');
    set(findobj(f,'type','line','Tag','Median'),'Color','g');
    set(findobj(f,'-regexp','Tag','\w*Whisker'),'LineStyle','-');
    set(sub_2,'TickLength',[0 0],'XTick',[],'XTickLabels',[]);

    figure_title('Distress Insurance Premium');
    
    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_scca(ds,id)

    el = sum(ds.SCCAExpectedLosses,2,'omitnan');
    cl = sum(ds.SCCAContingentLiabilities,2,'omitnan');
    alpha = cl ./ el;
    joint_es = ds.Indicators(:,6);

    f = figure('Name','Default Measures > Systemic CCA','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    sub_1 = subplot(2,2,[1 2]);
    a1 = area(sub_1,ds.DatesNum,smooth_data(el),'EdgeColor','none','FaceColor',[0.65 0.65 0.65]);
    hold on;
        p1 = plot(sub_1,ds.DatesNum,smooth_data(cl),'Color',[0.000 0.447 0.741]);
    hold off;
    legend(sub_1,[a1 p1],'Expected Losses','Contingent Liabilities','Location','best');
    t1 = title(sub_1,'Values');
    set(t1,'Units','normalized');
    t1_position = get(t1,'Position');
    set(t1,'Position',[0.4783 t1_position(2) t1_position(3)]);
    
    sub_2 = subplot(2,2,3);
    plot(sub_2,ds.DatesNum,smooth_data(alpha),'Color',[0.000 0.447 0.741]);
    t2 = title(sub_2,'Average Alpha');
    set(t2,'Units','normalized');
    t2_position = get(t2,'Position');
    set(t2,'Position',[0.4783 t2_position(2) t2_position(3)]);
    
    sub_3 = subplot(2,2,4);
    plot(sub_3,ds.DatesNum,smooth_data(joint_es),'Color',[0.000 0.447 0.741]);
    t3 = title(sub_3,['Joint ES (K=' sprintf('%.0f%%',(ds.K * 100)) ')']);
    set(t3,'Units','normalized');
    t3_position = get(t3,'Position');
    set(t3,'Position',[0.4783 t3_position(2) t3_position(3)]);
    
    set([sub_1 sub_2 sub_3],'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
    set([sub_2 sub_3],'XGrid','on','YGrid','on');

    if (ds.MonthlyTicks)
        date_ticks([sub_1 sub_2 sub_3],'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        date_ticks([sub_1 sub_2 sub_3],'x','yyyy','KeepLimits');
    end

    figure_title(['Systemic CCA (' ds.OP ')']);

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_sequence(ds,target,distance,id)

    n = ds.N;
    t = ds.T;
    dn = ds.DatesNum;
    mt = ds.MonthlyTicks;
    
    if (distance)
        ts = smooth_data(ds.(strrep(target,' ','')));
    else
        ts = smooth_data(ds.(strrep(target,' ','')));
    end
    
    data = [repmat({dn},1,n); mat2cell(ts,t,ones(1,n))];

    [~,index] = ismember(target,ds.LabelsSheetSimple);
    plots_title = repmat(ds.LabelsSheet(index),1,n);
    
    x_limits = [dn(1) dn(end)];

    if (distance)
        y_limits = plot_limits(ts,0.1,[],[],-1);
    else
        y_limits = plot_limits(ts,0.1);
    end

    core = struct();

    core.N = n;
    core.Data = data;
    core.Function = @(subs,data)plot_function(subs,data,distance);

    core.OuterTitle = ['Default Measures > ' target ' Time Series'];
    core.InnerTitle = [target ' Time Series'];
    core.SequenceTitles = ds.FirmNames;

    core.PlotsAllocation = [1 1];
    core.PlotsSpan = {1};
    core.PlotsTitle = plots_title;

    core.XDates = {mt};
    core.XGrid = {true};
    core.XLabel = {[]};
    core.XLimits = {x_limits};
    core.XRotation = {45};
    core.XTick = {[]};
    core.XTickLabels = {[]};

    core.YGrid = {true};
    core.YLabel = {[]};
    core.YLimits = {y_limits};
    core.YRotation = {[]};
    core.YTick = {[]};
    core.YTickLabels = {[]};

    sequential_plot(core,id);
    
    function plot_function(subs,data,distance)

        x = data{1};
        y = data{2};
        
        d = find(isnan(y),1,'first');
        
        if (isempty(d))
            xd = [];
        else
            xd = x(d) - 1;
        end
        
        plot(subs(1),x,y,'Color',[0.000 0.447 0.741]);
        
        if (distance)
            hold(subs(1),'on');
                plot(subs(1),x,zeros(numel(x),1),'Color',[1 0.4 0.4]);
            hold(subs(1),'off');
        end

        if (~isempty(xd))
            hold(subs(1),'on');
                plot(subs(1),[xd xd],get(subs(1),'YLim'),'Color',[1 0.4 0.4]);
            hold(subs(1),'off');
        end

    end

end
