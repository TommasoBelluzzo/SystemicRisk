% [INPUT]
% data = A structure representing the dataset.
% temp = A string representing the full path to the Excel spreadsheet used as a template for the results file.
% out = A string representing the full path to the Excel spreadsheet to which the results are written, eventually replacing the previous ones.
% bandwidth = An integer [21,252] representing the dimension of each rolling window (optional, default=252).
% steps = An integer [1,10] representing the number of steps between each rolling window (optional, default=10).
% lags = An integer [1,3] representing the number of lags of the VAR model in the variance decomposition (optional, default=2).
% h = An integer [1,10] representing the prediction horizon of the variance decomposition (optional, default=4).
% fevd = A string (either 'G' for Generalized or 'O' for Orthogonal) representing the FEVD type of the variance decomposition (optional, default='G').
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
        ip.addOptional('bandwidth',252,@(x)validateattributes(x,{'numeric'},{'scalar','integer','real','finite','>=',21,'<=',252}));
        ip.addOptional('steps',10,@(x)validateattributes(x,{'numeric'},{'scalar','integer','real','finite','>=',1,'<=',10}));
        ip.addOptional('lags',2,@(x)validateattributes(x,{'numeric'},{'scalar','integer','real','finite','>=',1,'<=',3}));
        ip.addOptional('h',4,@(x)validateattributes(x,{'numeric'},{'scalar','integer','real','finite','>=',1,'<=',10}));
        ip.addOptional('fevd','G',@(x)any(validatestring(x,{'G','O'})));
        ip.addOptional('analyze',false,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    data = validate_dataset(ipr.data,'spillover');
    temp = validate_template(ipr.temp);
    out = validate_output(ipr.out);
    indices = validate_rolling_windows(ipr.data,ipr.bandwidth,ipr.steps);
    
    nargoutchk(1,2);
    
    [result,stopped] = run_spillover_internal(data,temp,out,ipr.bandwidth,ipr.steps,indices,ipr.lags,ipr.h,ipr.fevd,ipr.analyze);

end

function [result,stopped] = run_spillover_internal(data,temp,out,bandwidth,steps,indices,lags,h,fevd,analyze)

    result = [];
    stopped = false;
    e = [];
    
    data = data_initialize(data,bandwidth,steps,indices,lags,h,fevd);

    rng_settings = rng();
    rng(0);
    
    bar = waitbar(0,'Initializing spillover measures...','CreateCancelBtn',@(src,event)setappdata(gcbf(),'Stop',true));
    setappdata(bar,'Stop',false);
    
    pause(1);
    waitbar(0,bar,'Calculating spillover measures...');
    pause(1);

    try

        windows_original = extract_rolling_windows(data.FirmReturns,bandwidth);
        windows = windows_original(indices);
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
    
    rng(rng_settings);
    
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
        try
            plot_index(data);
        catch
            warning('MATLAB:SystemicRisk','The analysis function ''plot_index'' produced errors.');
        end
        
        try
            plot_spillovers(data);
        catch
            warning('MATLAB:SystemicRisk','The analysis function ''plot_spillovers'' produced errors.');
        end
    end
    
    result = data;

end

%% DATA

function data = data_initialize(data,bandwidth,steps,indices,lags,h,fevd)

    w = numel(indices);

    data.Bandwidth = bandwidth;
    data.FEVD = fevd;
    data.H = h;
    data.Lags = lags;
    data.Steps = steps;
    data.Windows = w;
    data.WindowsIndices = indices;

    data.VarianceDecompositions = cell(w,1);
    data.SI = NaN(data.T,1);
    data.SpilloversFrom = NaN(data.T,data.N);
    data.SpilloversTo = NaN(data.T,data.N);
    data.SpilloversNet = NaN(data.T,data.N);

end

function data = data_finalize(data,window_results)

    n = data.N;
    t = data.T;

    bandwidth = data.Bandwidth;
    steps = data.Steps;
    windows_indices = data.WindowsIndices;

    index = 1;

    for i = windows_indices
        window_result = window_results{index};
        futures_result_off = bandwidth + i - 1;

        data.VarianceDecompositions{i} = window_result.VarianceDecomposition;
        data.SI(futures_result_off) = window_result.SI;
        data.SpilloversFrom(futures_result_off,:) = window_result.SpilloversFrom;
        data.SpilloversTo(futures_result_off,:) = window_result.SpilloversTo;
        data.SpilloversNet(futures_result_off,:) = window_result.SpilloversNet;
        
        index = index + 1;
    end
    
    if (steps > 1)
        x = 1:(t - bandwidth + 1);

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
            offset = bandwidth + k - 1;
            
            vd_k = vd{k};
            vd_k = bsxfun(@rdivide,vd_k,sum(vd_k,2));
            
            [si,spillovers_from,spillovers_to,spillovers_net] = calculate_spillover_measures(vd_k);
            
            data.VarianceDecompositions{k} = vd_k;
            data.SI(offset) = si;
            data.SpilloversFrom(offset,:) = spillovers_from;
            data.SpilloversTo(offset,:) = spillovers_to;
            data.SpilloversNet(offset,:) = spillovers_net;
        end
    end

end

function out = validate_output(out)

    [path,name,extension] = fileparts(out);

    if (~strcmp(extension,'.xlsx'))
        out = fullfile(path,[name extension '.xlsx']);
    end
    
end

function indices = validate_rolling_windows(data,bandwidth,steps)

    windows_count = data.T - bandwidth + 1;

    indices = 1:steps:windows_count;

    if (indices(end) ~= windows_count)
        indices = [indices windows_count];
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

function vd = variance_decomposition(window,lags,h,fevd) 

    n = size(window,2);

    zeros_indices = find(~window);
    window(zeros_indices) = (((1e-10 - 1e-8) .* rand(numel(zeros_indices),1)) + 1e-8);

    if (verLessThan('MATLAB','9.4'))
        spec = vgxset('n',n,'nAR',lags,'Constant',true);
        model = vgxvarx(spec,window(lags+1:end,:),[],window(1:lags,:));
        
        vma = vgxma(model,h,1:h);
        vma.MA(2:h+1) = vma.MA(1:h);
        vma.MA{1} = eye(n);
        
        covariance = vma.Q;
    else
        spec = varm(n,lags);
        model = estimate(spec,window(lags+1:end,:),'Y0',window(1:lags,:));

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
    
    if (strcmp(fevd,'G'))
        sigma = diag(covariance);

        for i = 1:n
            indices = zeros(n,1);
            indices(i,1) = 1;

            for j = 1:h
                irf(j,:,i) = (sigma(i,1).^-0.5) .* (vma.MA{j} * covariance * indices);
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

    irf_cs = cumsum(irf.^2);
    denominator = sum(irf_cs,3);

    for i = 1:n
        vds(:,:,i) = irf_cs(:,:,i) ./ denominator;     
    end

    vd = squeeze(vds(h,:,:));

end

%% PLOTTING

function plot_index(data)

    si = data.SI;
    si_max = max(si);
    si_max_sign = sign(si_max);

    f = figure('Name','Spillover Measures > Index','Units','normalized','Position',[100 100 0.85 0.85]);

    plot(data.DatesNum,si);
    ax = gca();

    y_limits = get(ax,'YLim');
    set(ax,'XLim',[data.DatesNum(data.Bandwidth) data.DatesNum(end)],'XTickLabelRotation',45,'YLim',[y_limits(1) ((abs(si_max) * 1.1) * si_max_sign)]);
    
    if (data.MonthlyTicks)
        datetick(ax,'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        datetick(ax,'x','yyyy','KeepLimits');
    end
    
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

function plot_spillovers(data)

    spillovers_from = data.SpilloversFrom;
    spillovers_from = bsxfun(@rdivide, spillovers_from, sum(spillovers_from,2));
    
    spillovers_to = data.SpilloversTo;
    spillovers_to = bsxfun(@rdivide, spillovers_to, sum(spillovers_to,2));

    spillovers_net = [min(data.SpilloversNet,[],2) max(data.SpilloversNet,[],2)];
    spillovers_net_avg = mean(spillovers_net,2);
    spillovers_net_avg(data.Bandwidth:end) = smooth(spillovers_net_avg(data.Bandwidth:end),'rlowess');

    f = figure('Name','Spillover Measures > Spillovers','Units','normalized','Position',[100 100 0.85 0.85]);

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
    fill(sub_3,[data.DatesNum(data.Bandwidth:end); flipud(data.DatesNum(data.Bandwidth:end))],[spillovers_net(data.Bandwidth:end,1); fliplr(spillovers_net(data.Bandwidth:end,2))],[0.65 0.65 0.65],'EdgeColor','none','FaceAlpha',0.35);
    hold on;
        plot(sub_3,data.DatesNum,spillovers_net_avg,'Color',[0.000 0.447 0.741]);
    hold off;
    set(sub_3,'YLim',[-1 1]);
    t4 = title(sub_3,'Net Spillovers');
    set(t4,'Units','normalized');
    t4_position = get(t4,'Position');
    set(t4,'Position',[0.4783 t4_position(2) t4_position(3)]);

    if (data.MonthlyTicks)
        datetick(sub_1,'x','mm/yyyy','KeepLimits','KeepTicks');
        datetick(sub_2,'x','mm/yyyy','KeepLimits','KeepTicks');
        datetick(sub_3,'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        datetick(sub_1,'x','yyyy','KeepLimits');
        datetick(sub_2,'x','yyyy','KeepLimits');
        datetick(sub_3,'x','yyyy','KeepLimits');
    end
    
    set([sub_1 sub_2 sub_3],'XLim',[data.DatesNum(data.Bandwidth) data.DatesNum(end)]);
    set([sub_1 sub_2],'YLim',[0 1],'YTick',0:0.2:1,'YTickLabels',arrayfun(@(x)sprintf('%.f%%',x),(0:0.2:1) * 100,'UniformOutput',false));
    set(sub_3,'YLim',[-1 1],'YTick',-1:0.2:1,'YTickLabels',arrayfun(@(x)sprintf('%.f%%',x),(-1:0.2:1) * 100,'UniformOutput',false));
    
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
