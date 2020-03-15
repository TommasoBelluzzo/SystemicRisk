% [INPUT]
% data = A structure representing the dataset.
% temp = A string representing the full path to the Excel spreadsheet used as a template for the results file.
% out = A string representing the full path to the Excel spreadsheet to which the results are written, eventually replacing the previous ones.
% bandwidth = An integer [21,252] representing the dimension of each rolling window (optional, default=252).
% rr = A float [0,1] representing the recovery rate in case of default (optional, default=0.4).
% lst = A float (0,INF) representing the long-term to short-term liabilities ratio used for the calculation of default barriers (optional, default=0.6).
% car = A float [0.03,0.20] representing the capital adequacy ratio used to calculate the D2C (optional, default=0.08).
% k = A float [0.90,0.99] representing the confidence level used within the Systemic CCA framework (optional, default=0.95).
% analyze = A boolean that indicates whether to analyse the results and display plots (optional, default=false).
%
% [OUTPUT]
% result = A structure representing the original dataset inclusive of intermediate and final calculations.
% stopped = A boolean that indicates whether the process has been stopped through user input.

function [result,stopped] = run_default(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data',@(x)validateattributes(x,{'struct'},{'nonempty'}));
        ip.addRequired('temp',@(x)validateattributes(x,{'char'},{'nonempty','size',[1 NaN]}));
        ip.addRequired('out',@(x)validateattributes(x,{'char'},{'nonempty','size',[1 NaN]}));
        ip.addOptional('bandwidth',252,@(x)validateattributes(x,{'numeric'},{'scalar','integer','real','finite','>=',21,'<=',252}));
        ip.addOptional('rr',0.4,@(x)validateattributes(x,{'double'},{'scalar','real','finite','>=',0,'<=',1}));
        ip.addOptional('lst',0.6,@(x)validateattributes(x,{'double'},{'scalar','real','finite','>',0}));
        ip.addOptional('car',0.08,@(x)validateattributes(x,{'double'},{'scalar','real','finite','>=',0.03,'<=',0.20}));
        ip.addOptional('k',0.95,@(x)validateattributes(x,{'double'},{'scalar','real','finite','>=',0.90,'<=',0.99}));
        ip.addOptional('analyze',false,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    data = validate_dataset(ipr.data,'default');
    temp = validate_template(ipr.temp);
    out = validate_output(ipr.out);
    
    nargoutchk(1,2);
    
    [result,stopped] = run_default_internal(data,temp,out,ipr.bandwidth,ipr.rr,ipr.lst,ipr.car,ipr.k,ipr.analyze);

end

function [result,stopped] = run_default_internal(data,temp,out,bandwidth,rr,lst,car,k,analyze)

    result = [];
    stopped = false;
    e = [];
    
    data = data_initialize(data,bandwidth,rr,lst,car,k);
    n = data.N;
    t = data.T;
    
    bar = waitbar(0,'Initializing default measures...','CreateCancelBtn',@(src,event)setappdata(gcbf(),'Stop',true));
    setappdata(bar,'Stop',false);
    
    pause(1);
    waitbar(0,bar,'Calculating default measures (step 1 of 2)...');
    pause(1);

    try

        r = max(0,data.RiskFreeRate);

        firms_data = extract_data_by_firm(data,{'Equity' 'Capitalization' 'Liabilities' 'CDS'});
        
        futures(1:n) = parallel.FevalFuture;
        futures_max = 0;
        futures_results = cell(n,1);

        for i = 1:n
            futures(i) = parfeval(@main_loop_1,1,firms_data{i},data.FirmDefaults(i),r,data.ST,data.DT,data.LCAR);
        end

        for i = 1:n
            if (getappdata(bar,'Stop'))
            	stopped = true;
                break;
            end
            
            [future_index,value] = fetchNext(futures);
            futures_results{future_index} = value;
            
            futures_max = max([future_index futures_max]);
            waitbar(0.2 * ((futures_max - 1) /  n),bar);

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
    waitbar(0.2,bar,'Finalizing default measures (step 1 of 2)...');
    pause(1);

    try
        data = data_finalize_1(data,futures_results);
    catch e
        delete(bar);
        rethrow(e);
    end
    
    pause(1);
    waitbar(0.2,bar,'Calculating default measures (step 2 of 2)...');
    pause(1);
    
    try

        cl = data.SCCAContingentLiabilities;
        cl(isnan(cl)) = 0;

        windows = extract_rolling_windows(cl,bandwidth,false);

        futures(1:t) = parallel.FevalFuture;
        futures_max = 0;
        futures_results = cell(t,1);

        for i = 1:t
            futures(i) = parfeval(@main_loop_2,1,windows{i},data.Q);
        end

        for i = 1:t
            if (getappdata(bar,'Stop'))
                stopped = true;
                break;
            end
            
            [future_index,value] = fetchNext(futures);
            futures_results{future_index} = value;
            
            futures_max = max([future_index futures_max]);
            waitbar(0.2 + (0.8 * ((futures_max - 1) / t)),bar);

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
        data = data_finalize_2(data,futures_results);
    catch e
        delete(bar);
        rethrow(e);
    end

    pause(1);
    waitbar(1,bar,'Writing default measures...');
	pause(1);
    
    try
        write_results(temp,out,data);
        delete(bar);
    catch e
        delete(bar);
        rethrow(e);
    end
    
    if (analyze)
        try
            plot_distances(data);
        catch
            warning('MATLAB:SystemicRisk','The analysis function ''plot_distances'' produced errors.');
        end
        
        try
            plot_scca(data);
        catch
            warning('MATLAB:SystemicRisk','The analysis function ''plot_scca'' produced errors.');
        end
    end
    
    result = data;

end

%% DATA

function data = data_initialize(data,bandwidth,rr,lst,car,k)

    q = [0.900:0.025:0.975 0.99];

    data.A = 1 - k;
    data.Bandwidth = bandwidth;
    data.CAR = car;
    data.DT = max(0.5,0.7 - (0.3 * (1 / lst)));
    data.K = k;
    data.LCAR = 1 / (1 - car);
    data.LGD = 1 - rr;
    data.LST = lst;
    data.Q = q(q >= k);
    data.RR = rr;
    data.ST =  1 / (1 + lst);

    car_label = sprintf('%.0f%%',(data.CAR * 100));
    lst_label = sprintf('%g',data.LST);
    data.LabelsIndicators = {'Average D2D' 'Average D2C' 'Portfolio D2D' 'Portfolio D2C' 'SCCA Joint ES'};
    data.LabelsSheet = {['D2D (LST=' lst_label ')'] ['D2C (LST=' lst_label ', CAR=' car_label ')'] 'SCCA Expected Losses' 'SCCA Contingent Liabilities' 'Indicators'};
    data.LabelsSheetSimple = {'D2D' 'D2C' 'SCCA Expected Losses' 'SCCA Contingent Liabilities' 'Indicators'};

    data.D2C = NaN(data.T,data.N);
    data.D2D = NaN(data.T,data.N);

    data.SCCAAlphas = NaN(data.T,data.N);
    data.SCCAExpectedLosses = NaN(data.T,data.N);
    data.SCCAContingentLiabilities = NaN(data.T,data.N);
    data.SCCAJointVaRs = NaN(data.T,numel(data.Q));

    data.Indicators = NaN(data.T,5);

end

function data = data_finalize_1(data,window_results)
  
    n = data.N;

    for i = 1:n
        window_result = window_results{i};

        data.D2D(1:window_result.Offset,i) = window_result.D2D;
        data.D2C(1:window_result.Offset,i) = window_result.D2C;

        data.SCCAAlphas(1:window_result.Offset,i) = window_result.SCCAAlphas;
        data.SCCAExpectedLosses(1:window_result.Offset,i) = window_result.SCCAExpectedLosses;
        data.SCCAContingentLiabilities(1:window_result.Offset,i) = window_result.SCCAContingentLiabilities;
    end

    weights = data.CapitalizationLagged ./ repmat(sum(data.CapitalizationLagged,2),1,n);

	cap = max(1e-6,sum(handle_defaulted_firms(data.Capitalization,data.FirmDefaults),2,'omitnan'));
    lb = max(1e-6,sum(handle_defaulted_firms(data.Liabilities,data.FirmDefaults),2,'omitnan'));
    db = (lb .* data.ST) + (data.DT .* (lb .* (1 - data.ST)));
    r = max(0,data.RiskFreeRate);

    [va,va_s] = kmv_model(cap,db,r,1);

    d2d_avg = sum(data.D2D .* weights,2,'omitnan');
    d2c_avg = sum(data.D2C .* weights,2,'omitnan');
	[d2d_por,d2c_por] = calculate_distances(va,va_s,db,r,1,data.LCAR);
    
    data.Indicators(:,1) = d2d_avg;
    data.Indicators(:,2) = d2c_avg;
    data.Indicators(:,3) = d2d_por;
    data.Indicators(:,4) = d2c_por;

end

function data = data_finalize_2(data,window_results)

    t = data.T;

    for i = 1:t
        window_result = window_results{i};
        data.SCCAJointVaRs(i,:) = window_result.SCCAJointVaRs;
        data.Indicators(i,5) = window_result.SCCAJointES;
    end

end

function out_file = validate_output(out_file)

    [path,name,extension] = fileparts(out_file);

    if (~strcmp(extension,'.xlsx'))
        out_file = fullfile(path,[name extension '.xlsx']);
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
        error(['The template must contain the following sheets: ' sheets{1} sprintf(', %s', sheets{2:end}) '.']);
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

function write_results(temp,out,data)

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

    dates_str = cell2table(data.DatesStr,'VariableNames',{'Date'});

    for i = 1:(numel(data.LabelsSheetSimple) - 1)
        sheet = data.LabelsSheetSimple{i};
        measure = strrep(sheet,' ','');

        tab = [dates_str array2table(data.(measure),'VariableNames',data.FirmNames)];
        writetable(tab,out,'FileType','spreadsheet','Sheet',sheet,'WriteRowNames',true);
    end

    tab = [dates_str array2table(data.Indicators,'VariableNames',strrep(data.LabelsIndicators,' ','_'))];
    writetable(tab,out,'FileType','spreadsheet','Sheet','Indicators','WriteRowNames',true);    

    if (ispc())
        try
            excel = actxserver('Excel.Application');
            exc_wb = excel.Workbooks.Open(out,0,false);

            for i = 1:numel(data.LabelsSheet)
                exc_wb.Sheets.Item(data.LabelsSheetSimple{i}).Name = data.LabelsSheet{i};
            end
            
            exc_wb.Save();
            exc_wb.Close();
            excel.Quit();
            
            delete(excel);
        catch
        end
    end

end

%% MEASURES

function window_results = main_loop_1(firm_data,firm_default,r,st,dt,lcar)

    offset = min(firm_default - 1,size(firm_data,1));

    eq = firm_data(1:offset,1);

    f = find(diff([false; eq < 0; false] ~= 0));
    indices = f(1:2:end-1);

    if (~isempty(indices))
        counts = f(2:2:end) - indices;

        index_last = indices(end);
        count_last = counts(end);

        if (((index_last + count_last - 1) == numel(eq)) && (count_last >= 252))
            offset = min(index_last - 1,offset);
        end
    end
    
    cap = max(1e-6,firm_data(1:offset,2));
    lb = max(1e-6,firm_data(1:offset,3));
    db = (lb .* st) + (dt .* (lb .* (1 - st)));
    r = r(1:offset);
    cds = firm_data(1:offset,4);
    
    [va,va_s] = kmv_model(cap,db,r,1);

    [d2d,d2c] = calculate_distances(va,va_s,db,r,1,lcar);
    [el,cl,a] = calculate_scca_values(va,va_s,db,r,cds,1);

    window_results.Offset = offset;
    
    window_results.D2D = d2d;
    window_results.D2C = d2c;

    window_results.SCCAAlphas = a;
    window_results.SCCAContingentLiabilities = cl;
    window_results.SCCAExpectedLosses = el;

end

function window_results = main_loop_2(window,q)

    window_results = struct();
    
    [scca_joint_vars,scca_joint_es] = calculate_scca_indicators(window,q);
    window_results.SCCAJointVaRs = scca_joint_vars;
    window_results.SCCAJointES = scca_joint_es;

end

function [d2d,d2c] = calculate_distances(va,va_s,db,r,t,lcar)

    rst = (r + (0.5 * va_s^2)) * t;
    st = va_s * sqrt(t);

    d1 = (log(va ./ db) + rst) ./ st;
    d2d = d1 - st;

    d1 = (log(va ./ (lcar .* db)) + rst) ./ st;
    d2c = d1 - st;

end

function [joint_vars,joint_es] = calculate_scca_indicators(data,q)

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
        v0 = unit_margin(x0,x0_mu,x0_sigma,x0_xi);
        
        e = [];

        try
            options = optimset(optimset(@fmincon),'Algorithm','sqp','Diagnostics','off','Display','off');
            [joint_var,~,ef] = fmincon(@(x)objective(x,v0,lhs,n,mu,sigma,xi),x0,[],[],[],[],[],[],[],options);
        catch e
        end

        if (~isempty(e) || (ef <= 0))
            joint_var = 0;
        end

        joint_vars(j) = joint_var;
    end
    
    q_diff = diff([q 1]);
    joint_es = sum(joint_vars .* q_diff) / sum(q_diff);
    
    function um = unit_margin(x,mu,sigma,xi)

        um_z = (x - mu) ./ sigma;
        um = (1 + (xi .* um_z)) .^ -(1 ./ xi);

    end

    function y = objective(x,v,lhs,n,mu,sigma,xi)

        ums = repelem(v,n,1);

        ums_check = (xi .* (repelem(x,20,1) - mu)) ./ sigma;
        ums_valid = isfinite(ums_check) & (ums_check > -1);

        ums(ums_valid) = unit_margin(repelem(x,sum(ums_valid),1),mu(ums_valid),sigma(ums_valid),xi(ums_valid));

        y = (sum(ums) - lhs)^2;

    end

end

function [el,cl,a] = calculate_scca_values(va,va_s,db,r,cds,t)

    dbd = db .* exp(-r.* t);
    st = va_s * sqrt(t);

    d1 = (log(va ./ db) + ((r + (0.5 * va_s^2)) .* t)) ./ st;
	d2 = d1 - st;

    put_price = (dbd .* normcdf(-d2)) - (va .* normcdf(-d1));
    put_price = max(0,put_price);

	rd = dbd - put_price;

    cds_put_price = dbd .* (1 - exp(-(cds ./ 10000) .* max(0.5,((db ./ rd) - 1)) .* t));
    cds_put_price = min(cds_put_price,put_price);  
    
    a = max(0,min(1 - (cds_put_price ./ put_price),1));
    a(~isreal(a)) = 0;
    
    el = put_price;
    cl = el .* a;

end

function [va,va_s] = kmv_model(eq,db,r,t)

    df = exp(-r.* t);

    k = numel(r);
    sk = sqrt(k);

    va = eq + (db .* df);
    va_r = diff(log(va));
    va_s = sqrt(252) * sqrt((1 / (k - 2)) * sum((va_r - mean(va_r)).^2));

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
        va_s = sqrt(252) * sqrt((1 / (k - 2)) * sum((va_r - mean(va_r)).^2));

        count = count + 1;
        error = norm(va - va_old) / sk;
    end

end

%% PLOTTING

function f = plot_distances(data)

    distances = data.Indicators(:,1:4);

    y_min = min(-1,min(min(distances)));
    y_max = max(max(distances(:,1)));
    y_limits = [((abs(y_min) * 1.1) * sign(y_min)) ((abs(y_max) * 1.1) * sign(y_max))];

    f = figure('Name','Default Measures > Distances','Units','normalized','Position',[100 100 0.85 0.85]);

    sub_1 = subplot(1,2,1);
    p1 = plot(sub_1,data.DatesNum,distances(:,1),'Color',[0.000 0.447 0.741]);
    hold on;
        p2 = plot(sub_1,data.DatesNum,distances(:,3),'Color',[0.494 0.184 0.556]);
        p3 = plot(sub_1,data.DatesNum,zeros(data.T,1),'Color',[1 0.4 0.4]);
    hold off;
    xlabel(sub_1,'Time');
    ylabel(sub_1,'Value');
    set(sub_1,'XLim',[data.DatesNum(1) data.DatesNum(end)],'YLim',y_limits,'XTickLabelRotation',45);
    title(sub_1,data.LabelsSheet{1});

    if (data.MonthlyTicks)
        datetick(sub_1,'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        datetick(sub_1,'x','yyyy','KeepLimits');
    end
    
    sub_2 = subplot(1,2,2);
    plot(sub_2,data.DatesNum,distances(:,2),'Color',[0.000 0.447 0.741]);
    hold on;
        plot(sub_2,data.DatesNum,distances(:,4),'Color',[0.494 0.184 0.556]);
        plot(sub_2,data.DatesNum,zeros(data.T,1),'Color',[1 0.4 0.4]);
    hold off;
    xlabel(sub_2,'Time');
    ylabel(sub_2,'Value');
    set(sub_2,'XLim',[data.DatesNum(1) data.DatesNum(end)],'YLim',y_limits,'XTickLabelRotation',45);
    title(sub_2,data.LabelsSheet{2});

    if (data.MonthlyTicks)
        datetick(sub_2,'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        datetick(sub_2,'x','yyyy','KeepLimits');
    end

    set([sub_1 sub_2],'YTick',get(sub_1,'YTick'),'YTickLabel',get(sub_1,'YTickLabel'));

    l = legend(sub_1,[p1 p2 p3],'Average','Portfolio','Default Threshold','Location','best');
    set(l,'NumColumns',3,'Units','normalized');
    drawnow();
    l_position = get(l,'Position');
    set(l,'Position',[0.4173 0.0328 l_position(3) l_position(4)]);

    t = figure_title('Distances');
    t_position = get(t,'Position');
    set(t,'Position',[t_position(1) -0.0157 t_position(3)]);

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function f = plot_scca(data)

    el = sum(data.SCCAExpectedLosses,2,'omitnan');
    cl = sum(data.SCCAContingentLiabilities,2,'omitnan');
    alpha = cl ./ el;
    jes = data.Indicators(:,5);

    f = figure('Name','Default Measures > Systemic CCA','Units','normalized','Position',[100 100 0.85 0.85]);

    sub_1 = subplot(2,2,[1 2]);
    a1 = area(sub_1,data.DatesNum,el,'EdgeColor','none','FaceColor',[0.65 0.65 0.65]);
    hold on;
        p1 = plot(sub_1,data.DatesNum,cl,'Color',[0.000 0.447 0.741]);
    hold off;
    xlabel(sub_1,'Time');
    ylabel(sub_1,'Value');
    legend(sub_1,[a1 p1],'Expected Losses','Contingent Liabilities','Location','best');
    t1 = title(sub_1,'Values');
    set(t1,'Units','normalized');
    t1_position = get(t1,'Position');
    set(t1,'Position',[0.4783 t1_position(2) t1_position(3)]);
    
    sub_2 = subplot(2,2,3);
    plot(sub_2,data.DatesNum,alpha,'Color',[0.000 0.447 0.741]);
    xlabel(sub_2,'Time');
    ylabel(sub_2,'Value');
    t2 = title(sub_2,'Average Alpha');
    set(t2,'Units','normalized');
    t2_position = get(t2,'Position');
    set(t2,'Position',[0.4783 t2_position(2) t2_position(3)]);
    
    sub_3 = subplot(2,2,4);
    plot(sub_3,data.DatesNum,jes,'Color',[0.000 0.447 0.741]);
    xlabel(sub_3,'Time');
    ylabel(sub_3,'Value');
    t3 = title(sub_3,['Joint ES (K=' sprintf('%.0f%%',(data.K * 100)) ')']);
    set(t3,'Units','normalized');
    t3_position = get(t3,'Position');
    set(t3,'Position',[0.4783 t3_position(2) t3_position(3)]);

    if (data.MonthlyTicks)
        datetick(sub_1,'x','mm/yyyy','KeepLimits','KeepTicks');
        datetick(sub_2,'x','mm/yyyy','KeepLimits','KeepTicks');
        datetick(sub_3,'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        datetick(sub_1,'x','yyyy','KeepLimits');
        datetick(sub_2,'x','yyyy','KeepLimits');
        datetick(sub_3,'x','yyyy','KeepLimits');
    end
    
    set([sub_1 sub_2 sub_3],'XLim',[data.DatesNum(1) data.DatesNum(end)],'XTickLabelRotation',45);

    t = figure_title('Systemic CCA');
    t_position = get(t,'Position');
    set(t,'Position',[t_position(1) -0.0157 t_position(3)]);

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end
