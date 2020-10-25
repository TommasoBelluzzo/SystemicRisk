 % [INPUT]
% ds = A structure representing the dataset.
% temp = A string representing the full path to the Excel spreadsheet used as a template for the results file.
% out = A string representing the full path to the Excel spreadsheet to which the results are written, eventually replacing the previous ones.
% bw = An integer [21,252] representing the dimension of each rolling window (optional, default=252).
% bws = An integer [1,10] representing the number of steps between each rolling window (optional, default=10).
% fevd = A string representing the FEVD type used by the variance decomposition (optional, default='G'):
%   - 'G' for generalized FEVD.
%   - 'O' for orthogonal FEVD.
% lags = An integer [1,3] representing the number of lags of the VAR model used by the variance decomposition (optional, default=2).
% h = An integer [1,10] representing the prediction horizon used by the variance decomposition (optional, default=4).
% analyze = A boolean that indicates whether to analyse the results and display plots (optional, default=false).
%
% [OUTPUT]
% result = A structure representing the original dataset inclusive of intermediate and final calculations.
% stopped = A boolean that indicates whether the process has been stopped through user input.

function [result,stopped] = run_spillover(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('ds',@(x)validateattributes(x,{'struct'},{'nonempty'}));
        ip.addRequired('temp',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addRequired('out',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addOptional('bw',252,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 21 '<=' 252 'scalar'}));
        ip.addOptional('bws',10,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 1 '<=' 10 'scalar'}));
        ip.addOptional('fevd','G',@(x)any(validatestring(x,{'G' 'O'})));
        ip.addOptional('lags',2,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 1 '<=' 3 'scalar'}));
        ip.addOptional('h',4,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 1 '<=' 10 'scalar'}));
        ip.addOptional('analyze',false,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    ds = validate_dataset(ipr.ds,'spillover');
    temp = validate_template(ipr.temp);
    out = validate_output(ipr.out);
    bw = ipr.bw;
    bws = ipr.bws;
    fevd = ipr.fevd;
    lags = ipr.lags;
    h = ipr.h;
    analyze = ipr.analyze;
    
    nargoutchk(1,2);
    
    [result,stopped] = run_spillover_internal(ds,temp,out,bw,bws,fevd,lags,h,analyze);

end

function [result,stopped] = run_spillover_internal(ds,temp,out,bw,bws,fevd,lags,h,analyze)

    result = [];
    stopped = false;
    e = [];

    indices = unique([1:bws:ds.T ds.T]);
    ds = initialize(ds,bw,bws,indices,fevd,lags,h);
    
    rng(double(bitxor(uint16('T'),uint16('B'))));
    cleanup_1 = onCleanup(@()rng('default'));
    
    bar = waitbar(0,'Initializing spillover measures...','CreateCancelBtn',@(src,event)setappdata(gcbf(),'Stop',true));
    setappdata(bar,'Stop',false);
    cleanup_2 = onCleanup(@()delete(bar));
    
    pause(1);
    waitbar(0,bar,'Calculating spillover measures...');
    pause(1);

    try

        windows = extract_rolling_windows(ds.Returns,ds.BW);
        windows = windows(indices);
        windows_len = numel(windows);

        futures(1:windows_len) = parallel.FevalFuture;
        futures_max = 0;
        futures_results = cell(windows_len,1);

        for i = 1:windows_len
           futures(i) = parfeval(@main_loop,1,windows{i},ds.Lags,ds.H,ds.FEVD);
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
        ds = finalize(ds,futures_results);
    catch e
        delete(bar);
        rethrow(e);
    end

    pause(1);
    waitbar(1,bar,'Writing spillover measures...');
    pause(1);
    
    try
        write_results(ds,temp,out);
        delete(bar);
    catch e
        delete(bar);
        rethrow(e);
    end

    if (analyze)
        safe_plot(@(id)plot_index(ds,id));
        safe_plot(@(id)plot_spillovers(ds,id));
        safe_plot(@(id)plot_sequence(ds,id));
    end
    
    result = ds;

end

%% PROCESS

function ds = initialize(ds,bw,bws,indices,fevd,lags,h)

    n = ds.N;
    t = ds.T;

    ds.BW = bw;
    ds.BWS = bws;
    ds.FEVD = fevd;
    ds.H = h;
    ds.Lags = lags;
    ds.Windows = numel(indices);
    ds.WindowsIndices = indices;

    all_label = [' (' fevd ', H=' num2str(ds.H) ', LAGS=' num2str(ds.Lags) ')'];

    ds.LabelsIndicatorsSimple = {'SI'};
    ds.LabelsIndicators = {['SI' all_label]};

    ds.LabelsSheetsSimple = {'From' 'To' 'Net' 'Indicators'};
    ds.LabelsSheets = {['From' all_label] ['To' all_label] ['Net' all_label] 'Indicators'};
    
    ds.VarianceDecompositions = cell(t,1);

    ds.SpilloversFrom = NaN(t,n);
    ds.SpilloversTo = NaN(t,n);
    ds.SpilloversNet = NaN(t,n);
    
    ds.Indicators = NaN(t,numel(ds.LabelsIndicators));
    
    ds.ComparisonReferences = {'Indicators' [] strcat({'SP-'},ds.LabelsIndicatorsSimple)};

end

function window_results = main_loop(r,lags,h,fevd)

    window_results = struct();

    vd = variance_decomposition(r,lags,h,fevd);
    window_results.VarianceDecomposition = vd;

    [sf,st,sn,si] = spillover_metrics(vd);
    window_results.SpilloversFrom = sf;
    window_results.SpilloversTo = st;
    window_results.SpilloversNet = sn;
    window_results.SI = si;

end

function ds = finalize(ds,results)

    window_indices = ds.WindowsIndices;

    for i = 1:numel(results)
        result = results{i};
        index = window_indices(i);

        ds.VarianceDecompositions{index} = result.VarianceDecomposition;
        
        ds.SpilloversFrom(index,:) = result.SpilloversFrom;
        ds.SpilloversTo(index,:) = result.SpilloversTo;
        ds.SpilloversNet(index,:) = result.SpilloversNet;
        
        ds.Indicators(index) = result.SI;
    end
    
    if (ds.BWS > 1)
        n = ds.N;
        t = ds.T;
        x = ds.DatesNum;

        nan_indices = ~ismember((1:t).',window_indices);
        step_indices = find(nan_indices);
        step_indices_len = numel(step_indices);

        vd = ds.VarianceDecompositions;
        vd(cellfun(@isempty,vd)) = {NaN(n,n)};

        for i = 1:n
            for j = 1:n
                vd_ij = cellfun(@(vdf)vdf(i,j),vd);
                vd_ij(nan_indices) = spline(x(~nan_indices),vd_ij(~nan_indices),x(nan_indices));

                for k = 1:step_indices_len
                    step_index = step_indices(k);

                    vd_k = vd{step_index};
                    vd_k(i,j) = vd_ij(step_index);
                    vd{step_index} = vd_k;
                end
            end   
        end

        for i = 1:step_indices_len
            step_index = step_indices(i);
            
            vd_i = vd{step_index};
            vd_i = bsxfun(@rdivide,vd_i,sum(vd_i,2));
            ds.VarianceDecompositions{step_index} = vd_i;

            [sf,st,sn,si] = spillover_metrics(vd_i);
            ds.SpilloversFrom(step_index,:) = sf;
            ds.SpilloversTo(step_index,:) = st;
            ds.SpilloversNet(step_index,:) = sn;
            ds.Indicators(step_index,1) = si;
        end
    end

    ds.SpilloversFrom = distress_data(ds.SpilloversFrom,ds.Defaults);
    ds.SpilloversTo = distress_data(ds.SpilloversTo,ds.Defaults);
    ds.SpilloversNet = distress_data(ds.SpilloversNet,ds.Defaults);

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

    for i = 1:(numel(ds.LabelsSheetsSimple) - 1)
        sheet = ds.LabelsSheetsSimple{i};
        measure = ['Spillovers' strrep(sheet,' ','')];

        tab = [dates_str array2table(ds.(measure),'VariableNames',ds.FirmNames)];
        writetable(tab,out,'FileType','spreadsheet','Sheet',sheet,'WriteRowNames',true);
    end
    
    tab = [dates_str array2table(ds.Indicators,'VariableNames',strrep(ds.LabelsIndicatorsSimple,' ','_'))];
    writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{end},'WriteRowNames',true);
    
    if (ispc())
        try
            excel = actxserver('Excel.Application');
        catch
            return;
        end

        try
            exc_wb = excel.Workbooks.Open(out,0,false);

            for i = 1:numel(ds.LabelsSheetsSimple)
                exc_wb.Sheets.Item(ds.LabelsSheetsSimple{i}).Name = ds.LabelsSheets{i};
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

%% PLOTTING

function plot_index(ds,id)

    si = smooth_data(ds.Indicators(:,1));

    f = figure('Name','Spillover Measures > Spillover Index','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    sub_1 = subplot(1,6,1:5);
    plot(sub_1,ds.DatesNum,si,'Color',[0.000 0.447 0.741]);
    set(sub_1,'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
    set(sub_1,'XGrid','on','YGrid','on');
    title(sub_1,ds.LabelsIndicators{1});
    
    if (ds.MonthlyTicks)
        date_ticks(sub_1,'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        date_ticks(sub_1,'x','yyyy','KeepLimits');
    end
    
    sub_2 = subplot(1,6,6);
    boxplot(sub_2,si,'Notch','on','Symbol','k.');
    set(findobj(f,'type','line','Tag','Median'),'Color','g');
    set(findobj(f,'-regexp','Tag','\w*Whisker'),'LineStyle','-');
    set(sub_2,'TickLength',[0 0],'XTick',[],'XTickLabels',[]);

	figure_title('Spillover Index');
    
    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_spillovers(ds,id)

    from = smooth_data(ds.SpilloversFrom);
    from = bsxfun(@rdivide,from,sum(from,2,'omitnan'));
    
    to = smooth_data(ds.SpilloversTo);
    to = bsxfun(@rdivide,to,sum(to,2,'omitnan'));

    net = smooth_data(ds.SpilloversNet);
    net = [min(net,[],2,'omitnan') max(net,[],2,'omitnan')];

    f = figure('Name','Spillover Measures > Spillovers','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    sub_1 = subplot(2,2,1);
    area(sub_1,ds.DatesNum,from,'EdgeColor',[0.000 0.447 0.741],'FaceColor','none');
    set(sub_1,'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
    set(sub_1,'YLim',[0 1],'YTick',0:0.2:1,'YTickLabels',arrayfun(@(x)sprintf('%.f%%',x),(0:0.2:1) .* 100,'UniformOutput',false));
    title(sub_1,'Spillovers From Others');

    sub_2 = subplot(2,2,3);
    area(sub_2,ds.DatesNum,to,'EdgeColor',[0.000 0.447 0.741],'FaceColor','none');
    set(sub_2,'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
    set(sub_2,'YLim',[0 1],'YTick',0:0.2:1,'YTickLabels',arrayfun(@(x)sprintf('%.f%%',x),(0:0.2:1) .* 100,'UniformOutput',false));
    title(sub_2,'Spillovers To Others');
    
    sub_3 = subplot(2,2,[2 4]);
    fill(sub_3,[ds.DatesNum; flipud(ds.DatesNum)],[net(:,1); fliplr(net(:,2))],[0.65 0.65 0.65],'EdgeColor','none','FaceAlpha',0.35);
    hold on;
        plot(sub_3,ds.DatesNum,mean(net,2),'Color',[0.000 0.447 0.741]);
        plot(sub_3,ds.DatesNum,zeros(ds.T,1),'Color',[1 0.4 0.4]);
    hold off;
    set(sub_3,'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
    set(sub_3,'YLim',[-1 1],'YTick',-1:0.2:1,'YTickLabels',arrayfun(@(x)sprintf('%.f%%',x),(-1:0.2:1) .* 100,'UniformOutput',false));
    set(sub_3,'XGrid','on','YGrid','on');
    title(sub_3,'Net Spillovers');

    if (ds.MonthlyTicks)
        date_ticks([sub_1 sub_2 sub_3],'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        date_ticks([sub_1 sub_2 sub_3],'x','yyyy','KeepLimits');
    end

    figure_title(['Spillovers (' ds.FEVD ', H=' num2str(ds.H) ', LAGS=' num2str(ds.Lags) ')']);
    
    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_sequence(ds,id)

    n = ds.N;
    t = ds.T;
    dn = ds.DatesNum;
    mt = ds.MonthlyTicks;
    
    from_all = smooth_data(ds.SpilloversFrom);
    to_all = smooth_data(ds.SpilloversTo);
    net_all = smooth_data(ds.SpilloversNet);

    data = [repmat({dn},1,n); mat2cell(from_all,t,ones(1,n)); mat2cell(to_all,t,ones(1,n)); mat2cell(net_all,t,ones(1,n))];
	
	plots_title = [repmat({'From'},1,n); repmat({'To'},1,n); repmat({'Net'},1,n)];
    
    x_limits = [dn(1) dn(end)];
    
    ft = [from_all to_all];
    y_limits_from = [0 1];
    y_limits_to = plot_limits(max(max(ft)),0.1,0);
    y_limits_to(2) = ceil(y_limits_to(2) * 10) / 10;
    y_limits_net = [-1 1];
    
    y_ticks_from = 0:0.2:1;
    y_ticks_to = 0:0.2:y_limits_to(2);
    y_ticks_net = -1:0.2:1;
    y_tick_labels = @(x)sprintf('%.f%%',x * 100);

    core = struct();

    core.N = n;
    core.Data = data;
    core.Function = @(subs,data)plot_function(subs,data);

    core.OuterTitle = 'Spillover Measures > Spillovers Time Series';
    core.InnerTitle = 'Spillovers Time Series';
    core.SequenceTitles = ds.FirmNames;

    core.PlotsAllocation = [3 1];
    core.PlotsSpan = {1 2 3};
    core.PlotsTitle = plots_title;

    core.XDates = {mt mt mt};
    core.XGrid = {true true true};
    core.XLabel = {[] [] []};
    core.XLimits = {x_limits x_limits x_limits};
    core.XRotation = {45 45 45};
    core.XTick = {[] [] []};
    core.XTickLabels = {[] [] []};

    core.YGrid = {true true true};
    core.YLabel = {[] [] []};
    core.YLimits = {y_limits_from y_limits_to y_limits_net};
    core.YRotation = {[] [] []};
    core.YTick = {y_ticks_from y_ticks_to y_ticks_net};
    core.YTickLabels = {y_tick_labels y_tick_labels y_tick_labels};
    
    sequential_plot(core,id);

    function plot_function(subs,data)

        x = data{1};
        from = data{2};
        to = data{3};
        net = data{4};
        
        d = find(isnan(net),1,'first');
        
        if (isempty(d))
            xd = [];
        else
            xd = x(d) - 1;
        end

        plot(subs(1),x,from,'Color',[0.000 0.447 0.741]);
        if (~isempty(xd))
            hold(subs(1),'on');
                plot(subs(1),[xd xd],get(subs(1),'YLim'),'Color',[1 0.4 0.4]);
            hold(subs(1),'off');
        end
        
        plot(subs(2),x,to,'Color',[0.000 0.447 0.741]);
        hold(subs(2),'on');
            plot(subs(2),x,ones(numel(x),1),'Color',[1 0.4 0.4]);
            if (~isempty(xd))
                plot(subs(2),[xd xd],get(subs(2),'YLim'),'Color',[1 0.4 0.4]);
            end
        hold(subs(2),'off');

        plot(subs(3),x,net,'Color',[0.000 0.447 0.741]);
        hold(subs(3),'on');
            plot(subs(3),x,zeros(numel(x),1),'Color',[1 0.4 0.4]);
            if (~isempty(xd))
                plot(subs(3),[xd xd],get(subs(3),'YLim'),'Color',[1 0.4 0.4]);
            end
        hold(subs(3),'off');

    end

end

%% VALIDATION

function out = validate_output(out)

    [path,name,extension] = fileparts(out);

    if (~strcmp(extension,'.xlsx'))
        out = fullfile(path,[name extension '.xlsx']);
    end
    
end

function temp = validate_template(temp)

    sheets = {'From' 'To' 'Net' 'Indicators'};
    file_sheets = validate_xls(temp,'T');

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
