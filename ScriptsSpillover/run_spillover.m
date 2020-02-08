% [INPUT]
% data = A structure representing the dataset.
% out_temp = A string representing the full path to the Excel spreadsheet used as a template for the results file.
% out_file = A string representing the full path to the Excel spreadsheet to which the results are written, eventually replacing the previous ones.
% rw_bandwidth = An integer greater than or equal to 30 representing the bandwidth (dimension) of each rolling window (optional, default=252).
% rw_steps = An integer [1,10] representing the number of steps between each rolling window (optional, default=10).
% lags = An integer [1,3] representing the number of lags of the VAR model in the variance decomposition (optional, default=2).
% h = An integer [1,10] representing the prediction horizon of the variance decomposition (optional, default=4).
% fevd = A string (either 'generalized or 'orthogonal') representing the FEVD type of the variance decomposition (optional, default='generalized').
% analyze = A boolean that indicates whether to analyse the results and display plots (optional, default=false).
%
% [OUTPUT]
% result = A structure representing the original dataset inclusive of intermediate and final calculations.

function result = run_spillover(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data',@(x)validateattributes(x,{'struct'},{'nonempty'}));
        ip.addRequired('out_temp',@(x)validateattributes(x,{'char'},{'nonempty','size',[1,NaN]}));
        ip.addRequired('out_file',@(x)validateattributes(x,{'char'},{'nonempty','size',[1,NaN]}));
        ip.addOptional('rw_bandwidth',252,@(x)validateattributes(x,{'numeric'},{'scalar','integer','real','finite','>=',30}));
        ip.addOptional('rw_steps',10,@(x)validateattributes(x,{'numeric'},{'scalar','integer','real','finite','>=',1,'<=',10}));
        ip.addOptional('lags',2,@(x)validateattributes(x,{'numeric'},{'scalar','integer','real','finite','>=',1,'<=',3}));
        ip.addOptional('h',4,@(x)validateattributes(x,{'numeric'},{'scalar','integer','real','finite','>=',1,'<=',10}));
        ip.addOptional('fevd','generalized',@(x)any(validatestring(x,{'generalized','orthogonal'})));
        ip.addOptional('analyze',false,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    data = validate_data(ipr.data);
    out_temp = validate_template(ipr.out_temp);
    out_file = validate_output(ipr.out_file);
    rw_indices = validate_rw_steps(ipr.data,ipr.rw_bandwidth,ipr.rw_steps);
    
    result = run_spillover_internal(data,out_temp,out_file,ipr.rw_bandwidth,ipr.rw_steps,rw_indices,ipr.lags,ipr.h,ipr.fevd,ipr.analyze);

end

function result = run_spillover_internal(data,out_temp,out_file,rw_bandwidth,rw_steps,rw_indices,lags,h,fevd,analyze)

    result = [];
    
    bar = waitbar(0,'Calculating spillover measures...','CreateCancelBtn',@(src,event)setappdata(gcbf(),'Stop',true));
    setappdata(bar,'Stop',false);
    
    windows_original = extract_rolling_windows(data.FirmReturns,rw_bandwidth);
    windows_original_len = length(windows_original);

    windows = windows_original(rw_indices);
    windows_len = length(windows);

    data = data_initialize(data,windows_original_len,rw_bandwidth,rw_steps,rw_indices,lags,h,fevd);

	rng_settings = rng();
    rng(0);
    
    stopped = false;
    e = [];

    futures(1:windows_len) = parallel.FevalFuture;
    futures_max = 0;
    futures_results = cell(windows_len,1);
    
    for i = 1:windows_len
       futures(i) = parfeval(@main_loop,1,windows{i},lags,h,fevd);
    end
    
    try
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
    
    data = data_finalize(data,futures_results);

    waitbar(100,bar,'Writing spillover measures...');
    write_results(out_temp,out_file,data);
    delete(bar);

    if (analyze)      
        plot_index(data);
        plot_spillovers(data);
    end
    
    result = data;

end

%% DATA

function data = data_initialize(data,windows_len,rw_bandwidth,rw_steps,rw_indices,lags,h,fevd)

    data.FEVD = fevd;
    data.H = h;
    data.Lags = lags;
    data.Windows = windows_len;
    data.WindowsBandwidth = rw_bandwidth;
    data.WindowsIndices = rw_indices;
    data.WindowsSteps = rw_steps;

    data.VarianceDecompositions = cell(windows_len,1);
    data.SI = NaN(data.T,1);
    data.SpilloversFrom = NaN(data.T,data.N);
    data.SpilloversTo = NaN(data.T,data.N);
    data.SpilloversNet = NaN(data.T,data.N);

end

function data = data_finalize(data,futures_results)

    index = 1;

    for i = data.WindowsIndices
        futures_result = futures_results{index};
        futures_result_off = data.WindowsBandwidth + i - 1;

        data.VarianceDecompositions{i} = futures_result.VarianceDecomposition;
        data.SI(futures_result_off) = futures_result.SI;
        data.SpilloversFrom(futures_result_off,:) = futures_result.SpilloversFrom;
        data.SpilloversTo(futures_result_off,:) = futures_result.SpilloversTo;
        data.SpilloversNet(futures_result_off,:) = futures_result.SpilloversNet;
        
        index = index + 1;
    end
    
    if (data.WindowsSteps > 1)
        x = 1:(data.T - data.WindowsBandwidth + 1);

        nans_check = ~ismember(x,data.WindowsIndices).';
        nans_indices = find(nans_check).';

        vd = data.VarianceDecompositions;
        vd(cellfun(@isempty,vd)) = {NaN(data.N,data.N)};

        for i = 1:data.N
            for j = 1:data.N
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
            offset = data.WindowsBandwidth + k - 1;
            
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

function data = validate_data(data)

    fields = {'Full', 'T', 'N', 'DatesNum', 'DatesStr', 'MonthlyTicks', 'IndexName', 'IndexReturns', 'FirmNames', 'FirmReturns', 'Capitalizations', 'CapitalizationsLagged', 'Liabilities', 'SeparateAccounts', 'StateVariables', 'Groups', 'GroupDelimiters', 'GroupNames'};
    
    for i = 1:numel(fields)
        if (~isfield(data,fields{i}))
            error('The dataset does not contain all the required data.');
        end
    end
    
end

function out_file = validate_output(out_file)

    [path,name,extension] = fileparts(out_file);

    if (~strcmp(extension,'.xlsx'))
        out_file = fullfile(path,[name extension '.xlsx']);
    end
    
end

function rw_indices = validate_rw_steps(data,rw_bandwidth,rw_steps)

    windows_count = data.T - rw_bandwidth + 1;

    rw_indices = 1:rw_steps:windows_count;

    if (rw_indices(end) ~= windows_count)
        rw_indices = [rw_indices, windows_count];
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

function write_results(out_temp,out_file,data)

    [out_file_path,~,~] = fileparts(out_file);

    if (exist(out_file_path,'dir') ~= 7)
        mkdir(out_file_path);
    end

    if (exist(out_file,'file') == 2)
        delete(out_file);
    end
    
    copy_result = copyfile(out_temp,out_file,'f');
    
    if (copy_result == 0)
        error('The results file could not be created from the template file.');
    end

    vars = [data.DatesStr num2cell(data.SI)];
    labels = {'Date' 'SI'};
    t1 = cell2table(vars,'VariableNames',labels);
    writetable(t1,out_file,'FileType','spreadsheet','Sheet','Index','WriteRowNames',true);

    vars = [data.DatesStr num2cell(data.SpilloversFrom)];
    labels = {'Date' data.FirmNames{:,:}};
    t2 = cell2table(vars,'VariableNames',labels);
    writetable(t2,out_file,'FileType','spreadsheet','Sheet','Spillovers From','WriteRowNames',true);

    vars = [data.DatesStr num2cell(data.SpilloversTo)];
    labels = {'Date' data.FirmNames{:,:}};
    t3 = cell2table(vars,'VariableNames',labels);
    writetable(t3,out_file,'FileType','spreadsheet','Sheet','Spillovers To','WriteRowNames',true);
    
    vars = [data.DatesStr num2cell(data.SpilloversNet)];
    labels = {'Date' data.FirmNames{:,:}};
    t4 = cell2table(vars,'VariableNames',labels);
    writetable(t4,out_file,'FileType','spreadsheet','Sheet','Spillovers Net','WriteRowNames',true);
    
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

%% PLOTTING

function plot_index(data)

    si = data.SI;
    si_max = max(si);
    si_max_sign = sign(si_max);

    f = figure('Name','Spillover Index','Units','normalized','Position',[100 100 0.85 0.85]);

    plot(data.DatesNum,si);
    ax = gca();

    y_limits = get(ax,'YLim');
    set(ax,'XLim',[data.DatesNum(data.WindowsBandwidth) data.DatesNum(end)],'XTickLabelRotation',45,'YLim',[y_limits(1) ((abs(si_max) * 1.1) * si_max_sign)]);
    
    if (data.MonthlyTicks)
        datetick(ax,'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        datetick(ax,'x','yyyy','KeepLimits');
    end
    
    if (strcmp(data.FEVD,'generalized'))
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
    spillovers_net_avg(data.WindowsBandwidth:end) = smooth(spillovers_net_avg(data.WindowsBandwidth:end),'rlowess');

    f = figure('Name','Spillovers','Units','normalized','Position',[100 100 0.85 0.85]);

    colors = get(gca(),'ColorOrder');
    color = colors(1,:);
    
    sub_1 = subplot(2,2,1);
    area(sub_1,data.DatesNum,spillovers_from,'EdgeColor',color,'FaceColor','none');
    t2 = title(sub_1,'Spillovers From Others');
    set(t2,'Units','normalized');
    t2_position = get(t2,'Position');
    set(t2,'Position',[0.4783 t2_position(2) t2_position(3)]);

    sub_2 = subplot(2,2,3);
    area(sub_2,data.DatesNum,spillovers_to,'EdgeColor',color,'FaceColor','none');
    t3 = title(sub_2,'Spillovers To Others');
    set(t3,'Units','normalized');
    t3_position = get(t3,'Position');
    set(t3,'Position',[0.4783 t3_position(2) t3_position(3)]);
    
    sub_3 = subplot(2,2,[2 4]);
    fill(sub_3,[data.DatesNum(data.WindowsBandwidth:end); flipud(data.DatesNum(data.WindowsBandwidth:end))],[spillovers_net(data.WindowsBandwidth:end,1); fliplr(spillovers_net(data.WindowsBandwidth:end,2))],[0.65 0.65 0.65],'EdgeColor','none','FaceAlpha',0.35);
    hold on;
        plot(sub_3,data.DatesNum,spillovers_net_avg,'Color',color);
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
    
    set([sub_1 sub_2 sub_3],'XLim',[data.DatesNum(data.WindowsBandwidth) data.DatesNum(end)]);
    set([sub_1 sub_2],'YLim',[0 1],'YTick',0:0.2:1,'YTickLabels',arrayfun(@(x)sprintf('%.f%%',x),(0:0.2:1) * 100,'UniformOutput',false));
    set(sub_3,'YLim',[-1 1],'YTick',-1:0.2:1,'YTickLabels',arrayfun(@(x)sprintf('%.f%%',x),(-1:0.2:1) * 100,'UniformOutput',false));
    
    if (strcmp(data.FEVD,'generalized'))
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
