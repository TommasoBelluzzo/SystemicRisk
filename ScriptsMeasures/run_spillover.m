% [INPUT]
% data = A structure representing the dataset.
% temp = A string representing the full path to the Excel spreadsheet used as a template for the results file.
% out = A string representing the full path to the Excel spreadsheet to which the results are written, eventually replacing the previous ones.
% bandwidth = An integer [21,252] representing the dimension of each rolling window (optional, default=252).
% steps = An integer [1,10] representing the number of steps between each rolling window (optional, default=10).
% lags = An integer [1,3] representing the number of lags of the VAR model in the variance decomposition (optional, default=2).
% h = An integer [1,10] representing the prediction horizon of the variance decomposition (optional, default=4).
% fevd = A string (either 'G' for generalized or 'O' for orthogonal) representing the FEVD type of the variance decomposition (optional, default='G').
% analyze = A boolean that indicates whether to analyse the results and display plots (optional, default=false).
%
% [OUTPUT]
% result = A structure representing the original dataset inclusive of intermediate and final calculations.
% stopped = A boolean that indicates whether the process has been stopped through user input.

function [result,stopped] = run_spillover(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data',@(x)validateattributes(x,{'struct'},{'nonempty'}));
        ip.addRequired('temp',@(x)validateattributes(x,{'char'},{'nonempty','size',[1,NaN]}));
        ip.addRequired('out',@(x)validateattributes(x,{'char'},{'nonempty','size',[1,NaN]}));
        ip.addOptional('bandwidth',252,@(x)validateattributes(x,{'double'},{'real','finite','integer','>=',21,'<=',252,'scalar'}));
        ip.addOptional('steps',10,@(x)validateattributes(x,{'double'},{'real','finite','integer','>=',1,'<=',10,'scalar'}));
        ip.addOptional('lags',2,@(x)validateattributes(x,{'double'},{'real','finite','integer','>=',1,'<=',3,'scalar'}));
        ip.addOptional('h',4,@(x)validateattributes(x,{'double'},{'real','finite','integer','>=',1,'<=',10,'scalar'}));
        ip.addOptional('fevd','G',@(x)any(validatestring(x,{'G','O'})));
        ip.addOptional('analyze',false,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    data = validate_dataset(ipr.data,'spillover');
    temp = validate_template(ipr.temp);
    out = validate_output(ipr.out);
    
    nargoutchk(1,2);
    
    [result,stopped] = run_spillover_internal(data,temp,out,ipr.bandwidth,ipr.steps,ipr.lags,ipr.h,ipr.fevd,ipr.analyze);

end

function [result,stopped] = run_spillover_internal(data,temp,out,bandwidth,steps,lags,h,fevd,analyze)

    result = [];
    stopped = false;
    e = [];

    indices = unique([1:steps:data.T data.T]);
    data = data_initialize(data,bandwidth,steps,indices,lags,h,fevd);
    
    rng(double(bitxor(uint16('T'),uint16('B'))));
	cleanup_1 = onCleanup(@()rng('default'));
    
    bar = waitbar(0,'Initializing spillover measures...','CreateCancelBtn',@(src,event)setappdata(gcbf(),'Stop',true));
    setappdata(bar,'Stop',false);
    cleanup_2 = onCleanup(@()delete(bar));
    
    pause(1);
    waitbar(0,bar,'Calculating spillover measures...');
    pause(1);

    try

        windows = extract_rolling_windows(data.Returns,bandwidth,false);
        windows = windows(indices);
        windows_len = numel(windows);

        futures(1:windows_len) = parallel.FevalFuture;
        futures_max = 0;
        futures_results = cell(windows_len,1);

        for i = 1:windows_len
           futures(i) = parfeval(@main_loop,1,windows{i},data.Lags,data.H,data.FEVD);
        end
        
        for i = 1:windows_len
            if (getappdata(bar,'Stop'))
                stopped = true;
                break;
            end
            
            [future_index,value] = fetchNext(futures);
            futures_results{future_index} = value;
            
            futures_max = max([future_index futures_max]);
            waitbar((futures_max - 1) / windows_len,bar);

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
    waitbar(1,bar,'Finalizing spillover measures...');
    pause(1);

    try
        data = data_finalize(data,futures_results);
    catch e
        delete(bar);
        rethrow(e);
    end

	pause(1);
    waitbar(1,bar,'Writing spillover measures...');
	pause(1);
    
    try
        write_results(temp,out,data);
        delete(bar);
    catch e
        delete(bar);
        rethrow(e);
    end

    if (analyze)
        safe_plot(@(id)plot_index(data,id));
        safe_plot(@(id)plot_spillovers(data,id));
        safe_plot(@(id)plot_sequence(data,id));
    end
    
    result = data;

end

%% DATA

function data = data_initialize(data,bandwidth,steps,indices,lags,h,fevd)

    data.Bandwidth = bandwidth;
    data.FEVD = fevd;
    data.H = h;
    data.Lags = lags;
    data.Steps = steps;
    data.Windows = numel(indices);
    data.WindowsIndices = indices;

    data.VarianceDecompositions = cell(data.T,1);
    data.SI = NaN(data.T,1);
    data.SpilloversFrom = NaN(data.T,data.N);
    data.SpilloversTo = NaN(data.T,data.N);
    data.SpilloversNet = NaN(data.T,data.N);

end

function data = data_finalize(data,results)

    n = data.N;
    t = data.T;

    steps = data.Steps;
    windows_indices = data.WindowsIndices;

    for i = 1:numel(results)
        window_result = results{i};
        window_offset = windows_indices(i);

        data.VarianceDecompositions{window_offset} = window_result.VarianceDecomposition;
        data.SI(window_offset) = window_result.SI;
        data.SpilloversFrom(window_offset,:) = window_result.SpilloversFrom;
        data.SpilloversTo(window_offset,:) = window_result.SpilloversTo;
        data.SpilloversNet(window_offset,:) = window_result.SpilloversNet;
    end

    if (steps > 1)
        x = 1:t;

        nans_check = ~ismember(x,windows_indices).';
        nans_indices = find(nans_check).';

        vd = data.VarianceDecompositions;
        vd(cellfun(@isempty,vd)) = {NaN(n,n)};

        for i = 1:n
            for j = 1:n
                vd_ij = cellfun(@(vdf)vdf(i,j),vd);
                vdij_spline = spline(x(~nans_check),vd_ij(~nans_check),x(nans_check));
                vd_ij(nans_check) = vdij_spline;

                for k = nans_indices
                    vd_k = vd{k};
                    vd_k(i,j) = vd_ij(k);
                    vd{k} = vd_k;
                end
            end   
        end

        for k = nans_indices
            vd_k = vd{k};
            vd_k = bsxfun(@rdivide,vd_k,sum(vd_k,2));
            
            [si,spillovers_from,spillovers_to,spillovers_net] = calculate_spillover_measures(vd_k);
            
            data.VarianceDecompositions{k} = vd_k;
            data.SI(k) = si;
            data.SpilloversFrom(k,:) = spillovers_from;
            data.SpilloversTo(k,:) = spillovers_to;
            data.SpilloversNet(k,:) = spillovers_net;
        end
    end
    
    data.SpilloversFrom = handle_firms_distress(data.Defaults,data.SpilloversFrom,false);
    data.SpilloversTo = handle_firms_distress(data.Defaults,data.SpilloversTo,false);
    data.SpilloversNet = handle_firms_distress(data.Defaults,data.SpilloversNet,false);

end

function out = validate_output(out)

    [path,name,extension] = fileparts(out);

    if (~strcmp(extension,'.xlsx'))
        out = fullfile(path,[name extension '.xlsx']);
    end
    
end

function temp = validate_template(temp)

    if (exist(temp,'file') == 0)
        error('The template file could not be found.');
    end
    
    if (ispc())
        [file_status,file_sheets,file_format] = xlsfinfo(temp);
        
        if (isempty(file_status) || ~strcmp(file_format,'xlOpenXMLWorkbook'))
            error('The dataset file is not a valid Excel spreadsheet.');
        end
    else
        [file_status,file_sheets] = xlsfinfo(temp);
        
        if (isempty(file_status))
            error('The dataset file is not a valid Excel spreadsheet.');
        end
    end
    
    sheets = {'Index' 'Spillovers From' 'Spillovers To' 'Spillovers Net'};

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

    vars = [data.DatesStr num2cell(data.SI)];
    labels = {'Date' 'SI'};
    t1 = cell2table(vars,'VariableNames',labels);
    writetable(t1,out,'FileType','spreadsheet','Sheet','Index','WriteRowNames',true);

    vars = [data.DatesStr num2cell(data.SpilloversFrom)];
    labels = {'Date' data.FirmNames{:,:}};
    t2 = cell2table(vars,'VariableNames',labels);
    writetable(t2,out,'FileType','spreadsheet','Sheet','Spillovers From','WriteRowNames',true);

    vars = [data.DatesStr num2cell(data.SpilloversTo)];
    labels = {'Date' data.FirmNames{:,:}};
    t3 = cell2table(vars,'VariableNames',labels);
    writetable(t3,out,'FileType','spreadsheet','Sheet','Spillovers To','WriteRowNames',true);
    
    vars = [data.DatesStr num2cell(data.SpilloversNet)];
    labels = {'Date' data.FirmNames{:,:}};
    t4 = cell2table(vars,'VariableNames',labels);
    writetable(t4,out,'FileType','spreadsheet','Sheet','Spillovers Net','WriteRowNames',true);
    
end

%% MEASURES

function window_results = main_loop(window,lags,h,fevd)

    window_results = struct();

    vd = variance_decomposition(window,lags,h,fevd);
    window_results.VarianceDecomposition = vd;

    [si,spillovers_from,spillovers_to,spillovers_net] = calculate_spillover_measures(vd);
    window_results.SI = si;
    window_results.SpilloversFrom = spillovers_from;
    window_results.SpilloversTo = spillovers_to;
    window_results.SpilloversNet = spillovers_net;

end

function [si,spillovers_from,spillovers_to,spillovers_net] = calculate_spillover_measures(vd)

    vd_diag = diag(vd);
    
    spillovers_from = sum(vd,2) - vd_diag;
    spillovers_to = sum(vd,1).' - vd_diag;
    spillovers_net = spillovers_to - spillovers_from;

    si = sum(spillovers_from,1) / (sum(vd_diag) + sum(spillovers_from,1));
    
    spillovers_from = spillovers_from.';
    spillovers_to = spillovers_to.';
    spillovers_net = spillovers_net.';

end

function c_hat = nearest_spd(c)

    a = (c + c.') ./ 2;
    [~,s,v] = svd(a);
    h = v * s * v.';

    c_hat = (a + h) ./ 2;
    c_hat = (c_hat + c_hat.') ./ 2;

	k = 0;
    p = 1;

    while (p ~= 0)
        [~,p] = chol(c_hat,'upper');
        k = k + 1;

        if (p ~= 0)
            e = min(eig(c_hat));
            c_hat = c_hat + (((-e * k^2) + eps(e)) * eye(size(c)));
        end
    end

end

function vd = variance_decomposition(data,lags,h,fevd) 

    [t,n] = size(data);
    d = max(n * 5,t) - t;
    k = t + d - lags;

    if (d > 0)
        mu = ones(d,1) .* mean(data,1);
        sigma = ones(d,1) .* std(data,1);

        rho = corr(data);
        rho(isnan(rho)) = 0;
        rho = nearest_spd(rho);

        z = (normrnd(mu,sigma,[d n]) * chol(rho,'upper')) + (0.01 .* randn(d,n));

        data = [data; z];
    end

    nan_indices = isnan(data);
    data(nan_indices) = 0;

    zero_indices = find(~data);
    data(zero_indices) = (-9e-9 .* rand(numel(zero_indices),1)) + 1e-8;

    novar_indices = find(var(data,1) == 0);
    data(:,novar_indices) = data(:,novar_indices) + ((-9e-9 .* rand(size(data(:,novar_indices)))) + 1e-8);

    e = [data(1:lags,:); data(lags+1:end,:)];

    ar_first = n + 1;
    ar_start = (lags * n^2) + ar_first;
    trend = ar_start:(ar_start+n-1);

    params = (lags * n^2) + (2 * n);
    f = NaN(params,1);
    f(trend) = zeros(n,1);
    fs = true(params,1);
    fs(trend) = false;

    z = zeros(n,params);
    z(:,1:n) = eye(n);

    x = cell(k,1);
    y = e(lags+1:end,:);

    for t = (lags + 1):(lags + k)
        ar_start = ar_first;
        ar_x = t - lags;

        for i = 1:lags
            indices = ar_start:(ar_start + n^2 - 1);
            z(:,indices) = kron(e(t-i,:),eye(n));
            ar_start = indices(end) + 1;
        end

        z(:,trend) = ar_x * eye(n);
        x{ar_x} = z(:,fs);
        y(ar_x,:) = y(ar_x,:) - (z(:,~fs) * f(~fs)).';
    end

    [f(fs),covariance] = mvregress(x,y,'CovType','full','VarFormat','beta','VarType','fisher','MaxIter',1000);

    coefficients = cell(1,lags);
    ar_start = ar_first;

    for i = 1:lags
        indices = ar_start:(ar_start + n^2 - 1);
        coefficients{i} = reshape(f(indices),n,n);
        ar_start = indices(end) + 1;
    end

    r = zeros(n * lags,n * lags);
    r(1:n,:) = cell2mat(coefficients);

    if (lags > 2)
        r(n+1:end,1:(end-n)) = eye((lags - 1) * n);
    end

    ma = cell(h,1);
    ma{1} = eye(n);
    ma{2} = r(1:n,1:n);

    if (h >= 3)
        for i = 3:h
            temp = r^i;
            ma{i} = temp(1:n,1:n);
        end
    end
    
    irf = zeros(h,n,n);
    vds = zeros(h,n,n);
    
    if (strcmp(fevd,'G'))
        sigma = diag(covariance);

        for i = 1:n
            indices = zeros(n,1);
            indices(i,1) = 1;

            for j = 1:h
                irf(j,:,i) = (sigma(i,1) .^ -0.5) .* (ma{j} * covariance * indices);
            end
        end
    else
        covariance = nearest_spd(covariance);
        covariance_dec = chol(covariance,'lower');

        for i = 1:n
            indices = zeros(n,1);
            indices(i,1) = 1;

            for j = 1:h
                irf(j,:,i) = ma{j} * covariance_dec * indices; 
            end
        end
    end

    irf_cs = cumsum(irf.^2);
    irf_cs_sum = sum(irf_cs,3);

    for i = 1:n
        vds(:,:,i) = irf_cs(:,:,i) ./ irf_cs_sum;     
    end

    vd = squeeze(vds(h,:,:));

end

%% PLOTTING

function plot_index(data,id)

    f = figure('Name','Spillover Measures > Index','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    sub_1 = subplot(1,6,1:5);
    plot(sub_1,data.DatesNum,data.SI);
    set(sub_1,'XLim',[data.DatesNum(1) data.DatesNum(end)],'XTickLabelRotation',45);
    
    if (data.MonthlyTicks)
        datetick(sub_1,'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        datetick(sub_1,'x','yyyy','KeepLimits');
    end
    
    sub_2 = subplot(1,6,6);
    boxplot(sub_2,data.SI,'Notch','on','Symbol','k.');
    set(findobj(f,'type','line','Tag','Median'),'Color','g');
    set(findobj(f,'-regexp','Tag','\w*Whisker'),'LineStyle','-');
    delete(findobj(f,'-regexp','Tag','\w*Outlier'));
    set(sub_2,'TickLength',[0 0],'XTick',[],'XTickLabels',[]);
    
    if (strcmp(data.FEVD,'G'))
        t = figure_title('Spillover Index (Generalized FEVD)');
    else
        t = figure_title('Spillover Index (Orthogonal FEVD)');
    end

    t_position = get(t,'Position');
    set(t,'Position',[t_position(1) -0.0157 t_position(3)]);
    
    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_spillovers(data,id)

    spillovers_from = data.SpilloversFrom;
    spillovers_from = bsxfun(@rdivide,spillovers_from,sum(spillovers_from,2,'omitnan'));
    
    spillovers_to = data.SpilloversTo;
    spillovers_to = bsxfun(@rdivide,spillovers_to,sum(spillovers_to,2,'omitnan'));

    spillovers_net = data.SpilloversNet;
    spillovers_net = [min(spillovers_net,[],2,'omitnan') max(spillovers_net,[],2,'omitnan')];
    spillovers_net_avg = mean(spillovers_net,2);

    f = figure('Name','Spillover Measures > Spillovers','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    sub_1 = subplot(2,2,1);
    area(sub_1,data.DatesNum,spillovers_from,'EdgeColor',[0.000 0.447 0.741],'FaceColor','none');
    t2 = title(sub_1,'Spillovers From Others');
    set(t2,'Units','normalized');
    t2_position = get(t2,'Position');
    set(t2,'Position',[0.4783 t2_position(2) t2_position(3)]);

    sub_2 = subplot(2,2,3);
    area(sub_2,data.DatesNum,spillovers_to,'EdgeColor',[0.000 0.447 0.741],'FaceColor','none');
    t3 = title(sub_2,'Spillovers To Others');
    set(t3,'Units','normalized');
    t3_position = get(t3,'Position');
    set(t3,'Position',[0.4783 t3_position(2) t3_position(3)]);
    
    sub_3 = subplot(2,2,[2 4]);
    fill(sub_3,[data.DatesNum; flipud(data.DatesNum)],[spillovers_net(:,1); fliplr(spillovers_net(:,2))],[0.65 0.65 0.65],'EdgeColor','none','FaceAlpha',0.35);
    hold on;
        plot(sub_3,data.DatesNum,spillovers_net_avg,'Color',[0.000 0.447 0.741]);
    hold off;
    grid on;
    set(sub_3,'YLim',[-1 1]);
    t4 = title(sub_3,'Net Spillovers');
    set(t4,'Units','normalized');
    t4_position = get(t4,'Position');
    set(t4,'Position',[0.4783 t4_position(2) t4_position(3)]);
    
    set([sub_1 sub_2 sub_3],'XLim',[data.DatesNum(1) data.DatesNum(end)],'XTickLabelRotation',45);
    set([sub_1 sub_2],'YLim',[0 1],'YTick',0:0.2:1,'YTickLabels',arrayfun(@(x)sprintf('%.f%%',x),(0:0.2:1) * 100,'UniformOutput',false));
    set(sub_3,'YLim',[-1 1],'YTick',-1:0.2:1,'YTickLabels',arrayfun(@(x)sprintf('%.f%%',x),(-1:0.2:1) * 100,'UniformOutput',false));

    if (data.MonthlyTicks)
        datetick(sub_1,'x','mm/yyyy','KeepLimits','KeepTicks');
        datetick(sub_2,'x','mm/yyyy','KeepLimits','KeepTicks');
        datetick(sub_3,'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        datetick(sub_1,'x','yyyy','KeepLimits');
        datetick(sub_2,'x','yyyy','KeepLimits');
        datetick(sub_3,'x','yyyy','KeepLimits');
    end

    if (strcmp(data.FEVD,'G'))
        t = figure_title('Spillovers (Generalized FEVD)');
    else
        t = figure_title('Spillovers (Orthogonal FEVD)');
    end
    
    t_position = get(t,'Position');
    set(t,'Position',[t_position(1) -0.0157 t_position(3)]);
    
    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_sequence(data,id)

    ft = [data.SpilloversFrom data.SpilloversTo];

    x = data.DatesNum;
    x_limits = [x(1) x(end)];
    
    y = cat(3,data.SpilloversFrom,data.SpilloversTo,data.SpilloversNet);
    
    y_min_ft = min(min(ft));
    y_max_ft = max(max(ft));
    y_min_net = min(min(data.SpilloversNet));
    y_max_net = max(max(data.SpilloversNet));
    
    y_limits = zeros(3,2);
    y_limits(1:2,:) = repmat([((abs(y_min_ft) * 1.1) * sign(y_min_ft)) ((abs(y_max_ft) * 1.1) * sign(y_max_ft))],2,1);
    y_limits(3,:) = [((abs(y_min_net) * 1.1) * sign(y_min_net)) ((abs(y_max_net) * 1.1) * sign(y_max_net))];

    core = struct();

    core.N = data.N;
    core.PlotFunction = @(ax,x,y)plot_function(ax,x,y);
    core.SequenceFunction = @(y,offset)[y(:,offset,1) y(:,offset,2) y(:,offset,3)];
	
    core.OuterTitle = 'Cross-Sectional Measures';
    core.InnerTitle = 'Spillovers Time Series';
    core.Labels = data.FirmNames;

    core.Plots = 3;
    core.PlotsTitle = {'From' 'To' 'Net'};
    core.PlotsType = 'V';
    
    core.X = x;
    core.XDates = data.MonthlyTicks;
    core.XLabel = 'Time';
    core.XLimits = x_limits;
    core.XRotation = 45;
    core.XTick = [];
    core.XTickLabels = @(x)sprintf('%.2f',x);

    core.Y = y;
    core.YLabel = 'Value';
    core.YLimits = y_limits;
    core.YRotation = [];
    core.YTick = [];
    core.YTickLabels = [];

    sequential_plot(core,id);

    function plot_function(subs,x,y)

        for i = 1:3
            plot(subs(i),x,y(:,i),'Color',[0.000 0.447 0.741]);
        end

    end

end
