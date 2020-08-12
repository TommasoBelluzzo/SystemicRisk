% [INPUT]
% ds = A structure representing the dataset.
% temp = A string representing the full path to the Excel spreadsheet used as a template for the results file.
% out = A string representing the full path to the Excel spreadsheet to which the results are written, eventually replacing the previous ones.
% sel = A string (either 'A' for automatic, 'F' for forced firms or 'G' for forced groups) representing the time series selection method (optional, default='A').
% bw = An integer [21,252] representing the dimension of each rolling window (optional, default=252).
% rr = A float [0,1] representing the recovery rate in case of default (optional, default=0.4).
% pw = A string (either 'A' for plain average or 'W' for progressive average) representing the probabilities of default averaging method (optional, default='W').
% md = A string (either 'N' for normal or 'T' for Student's T) representing the multivariate distribution used by the CIMDO model (optional, default='N').
% analyze = A boolean that indicates whether to analyse the results and display plots (optional, default=false).
%
% [OUTPUT]
% result = A structure representing the original dataset inclusive of intermediate and final calculations.
% stopped = A boolean that indicates whether the process has been stopped through user input.

function [result,stopped] = run_cross_entropy(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('ds',@(x)validateattributes(x,{'struct'},{'nonempty'}));
        ip.addRequired('temp',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addRequired('out',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addOptional('sel','A',@(x)any(validatestring(x,{'A' 'F' 'G'})));
        ip.addOptional('bw',252,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 21 '<=' 252 'scalar'}));
        ip.addOptional('rr',0.4,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0 '<=' 1 'scalar'}));
        ip.addOptional('pw','W',@(x)any(validatestring(x,{'A' 'W'})));
        ip.addOptional('md','N',@(x)any(validatestring(x,{'N' 'T'})));
        ip.addOptional('analyze',false,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    ds = validate_dataset(ipr.ds,'cross-entropy');
    temp = validate_template(ipr.temp);
    out = validate_output(ipr.out);
    sel = validate_selection(ipr.sel,ds.N,ds.Groups);
    bw = ipr.bw;
    rr = ipr.rr;
    pw = ipr.pw;
    md = ipr.md;
    analyze = ipr.analyze;
    
    nargoutchk(1,2);
    
    [result,stopped] = run_cross_entropy_internal(ds,temp,out,sel,bw,rr,pw,md,analyze);

end

function [result,stopped] = run_cross_entropy_internal(ds,temp,out,sel,bw,rr,pw,md,analyze)

    result = [];
    stopped = false;
    e = [];
    
    ds = initialize(ds,sel,bw,rr,pw,md);
    t = ds.T;
    
    bar = waitbar(0,'Initializing cross-entropy measures...','CreateCancelBtn',@(src,event)setappdata(gcbf(),'Stop',true));
    setappdata(bar,'Stop',false);
    cleanup = onCleanup(@()delete(bar));

    pause(1);
    waitbar(0,bar,'Calculating cross-entropy measures...');
    pause(1);

    try

        windows_r = extract_rolling_windows(ds.TargetReturns,ds.BW);
        windows_pods = extract_rolling_windows(ds.TargetPoDs,ds.BW);

        futures(1:t) = parallel.FevalFuture;
        futures_max = 0;
        futures_results = cell(t,1);

        for i = 1:t
            futures(i) = parfeval(@main_loop,1,windows_r{i},windows_pods{i},ds.PW,ds.MD);
        end

        for i = 1:t
            if (getappdata(bar,'Stop'))
                stopped = true;
                break;
            end
            
            [future_index,value] = fetchNext(futures);
            futures_results{future_index} = value;
            
            futures_max = max([future_index futures_max]);
            waitbar((futures_max - 1) / t,bar);

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
    waitbar(1,bar,'Finalizing cross-entropy measures...');
    pause(1);

    try
        ds = finalize(ds,futures_results);
    catch e
        delete(bar);
        rethrow(e);
    end

    pause(1);
    waitbar(1,bar,'Writing cross-entropy measures...');
    pause(1);
    
    try
        write_results(ds,temp,out);
        delete(bar);
    catch e
        delete(bar);
        rethrow(e);
    end

    if (analyze)
        safe_plot(@(id)plot_indicators(ds,id));
        safe_plot(@(id)plot_dide(ds,id));
        safe_plot(@(id)plot_sequence_dide(ds,id));
        safe_plot(@(id)plot_sequence_cojpods(ds,id));
    end
    
    result = ds;

end

%% DATA

function ds = initialize(ds,sel,bw,rr,pw,md)

    n = ds.N;
    t = ds.T;
    
    by_groups = (strcmp(sel,'A') && (n >= 10)) || strcmp(sel,'G');

    if (by_groups)
        g = ds.Groups;
        gd = ds.GroupDelimiters;

        pods_ref = ds.CDS ./ rr;
        pods = zeros(t,g);
        
        r_ref = ds.Returns;
        r = zeros(t,g);
        
        cp = ds.Capitalizations;

        if (isempty(cp))
            for i = 1:g
                if (i == 1)
                    seq = 1:gd(1);
                elseif (i == g)
                    seq = (gd(i - 1) + 1):n;
                else
                    seq = (gd(i - 1) + 1):gd(i);
                end

                sequence_len = numel(seq);

                r_i = r_ref(:,seq);
                w_i = 1 ./ (repmat(sequence_len,t,1) - sum(isnan(r_i),2));
                r(:,i) = sum(r_i .* repmat(w_i,1,sequence_len),2,'omitnan');

                pods_i = pods_ref(:,seq);
                w_i = 1 ./ (repmat(sequence_len,t,1) - sum(isnan(pods_i),2));
                pods(:,i) = sum(pods_i .* repmat(w_i,1,sequence_len),2,'omitnan');
            end
        else
            for i = 1:g
                if (i == 1)
                    seq = 1:gd(1);
                elseif (i == g)
                    seq = (gd(i - 1) + 1):n;
                else
                    seq = (gd(i - 1) + 1):gd(i);
                end

                sequence_len = numel(seq);
                
                cp_i = cp(:,seq);
                w_i = max(0,cp_i ./ repmat(sum(cp_i,2,'omitnan'),1,sequence_len));

                r(:,i) = sum(r_ref(:,seq) .* w_i,2,'omitnan');
                pods(:,i) = sum(pods_ref(:,seq) .* w_i,2,'omitnan');
            end
        end

        n = g;
    else
        r = ds.Returns;
        pods = ds.CDS ./ rr;
    end

    ds.BW = bw;
    ds.ByGroups = by_groups;
    ds.LGD = 1 - rr;
    ds.MD = md;
    ds.PW = pw;
    ds.RR = rr;
    ds.TargetPoDs = pods;
    ds.TargetReturns = r;

    ds.LabelsIndicators = {'JPoD' 'FSI' 'PCE'};
    ds.LabelsSheet = {'Indicators' 'CoJPoDs' 'Systemic Importance' 'Systemic Vulnerability'};

    ds.Indicators = NaN(t,numel(ds.LabelsIndicators));
    
    ds.AverageDiDe = NaN(n);
    ds.DiDe = cell(t,1);
    ds.SI = NaN(t,n);
    ds.SV = NaN(t,n);
    
    ds.CoJPoDs = NaN(t,n);

end

function ds = finalize(ds,results)

    t = ds.T;

    for i = 1:t
        window_result = results{i};

        ds.Indicators(i,:) = [window_result.JPoD window_result.FSI window_result.PCE];
        
        ds.DiDe{i} = window_result.DiDe;
        ds.SI(i,:) = window_result.SI;
        ds.SV(i,:) = window_result.SV;

        ds.CoJPoDs(i,:) = window_result.CoJPoDs;
    end
    
    ds.AverageDiDe = sum(cat(3,ds.DiDe{:}),3) ./ numel(ds.DiDe);

    si = ds.SI;
    si_max = max(max(si,[],'omitnan'),[],'omitnan');
    si_min = min(min(si,[],'omitnan'),[],'omitnan');
    ds.SI = (si - si_min) ./ (si_max - si_min);
    
    sv = ds.SV;
    sv_max = max(max(sv,[],'omitnan'),[],'omitnan');
    sv_min = min(min(sv,[],'omitnan'),[],'omitnan');
    ds.SV = (sv - sv_min) ./ (sv_max - sv_min);

end

function out = validate_output(out)

    [path,name,extension] = fileparts(out);

    if (~strcmp(extension,'.xlsx'))
        out = fullfile(path,[name extension '.xlsx']);
    end
    
end

function sel = validate_selection(sel,n,groups)

    if (strcmp(sel,'F') && (n > 10))
        error('The selection cannot be forced to firms because their number is greater than 10.');
    end
    
    if (strcmp(sel,'G'))
        if (groups == 0)
            error('The selection cannot be forced to groups because their are not defined.');
        end
        
        if (groups > 10)
            error('The selection cannot be forced to groups because their number is greater than 10.');
        end
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
    
    sheets = {'Indicators' 'Average DiDe' 'SI' 'SV' 'CoJPoDs'};

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
        error('The results file could not be created from the template file.');
    end

    dates_str = cell2table(ds.DatesStr,'VariableNames',{'Date'});

	labels = strrep(ds.LabelsIndicators,'-','_');
    t1 = [dates_str array2table(ds.Indicators,'VariableNames',labels)];
    writetable(t1,out,'FileType','spreadsheet','Sheet','Indicators','WriteRowNames',true);

    if (ds.ByGroups)
        header = 'Groups';
        labels = ds.GroupShortNames;
    else
        header = 'Firms';
        labels = ds.FirmNames;
    end
    
    vars = [labels num2cell(ds.AverageDiDe)];
    t2 = cell2table(vars,'VariableNames',{header labels{:,:}});
    writetable(t2,out,'FileType','spreadsheet','Sheet','Average DiDe','WriteRowNames',true);

    t3 = [dates_str array2table(ds.SI,'VariableNames',labels)];
    writetable(t3,out,'FileType','spreadsheet','Sheet','SI','WriteRowNames',true);

    t4 = [dates_str array2table(ds.SV,'VariableNames',labels)];
    writetable(t4,out,'FileType','spreadsheet','Sheet','SV','WriteRowNames',true);
    
    t5 = [dates_str array2table(ds.CoJPoDs,'VariableNames',labels)];
    writetable(t5,out,'FileType','spreadsheet','Sheet','CoJPoDs','WriteRowNames',true);
    
end

%% MEASURES

function window_results = main_loop(r,pods,pw,md)

    window_results = struct();

    nan_indices = any(isnan(r),1);
    n = numel(nan_indices);
    
    r(:,nan_indices) = [];
    pods(:,nan_indices) = [];

    if (strcmp(pw,'A'))
        pods = mean(pods,1).';
    else
        [t,n] = size(pods);
        w = repmat(fliplr(((1 - 0.98) / (1 - 0.98^t)) .* (0.98 .^ (0:1:t-1))).',1,n);
        pods = sum(pods .* w,1).';
    end

	[g,p] = cimdo(r,pods,md);

    if (any(isnan(p)))
        window_results.JPoD = NaN;
        window_results.FSI = NaN;
        window_results.PCE = NaN;
        
        window_results.DiDe = NaN(n);
        window_results.SI = NaN(1,n);
        window_results.SV = NaN(1,n);
        
        window_results.CoJPoDs = NaN(1,n);
    else
        opods = pods;
        pods = NaN(n,1);
        pods(~nan_indices) = opods;
        
        g_refs = sum(g,2);
        
        [jpod,fsi,pce] = calculate_indicators(n,pods,g_refs,p);
        window_results.JPoD = jpod;
        window_results.FSI = fsi;
        window_results.PCE = pce;

        [dide,si,sv] = calculate_dide(n,pods,g,g_refs,p);
        window_results.DiDe = dide;
        window_results.SI = si;
        window_results.SV = sv;
        
        cojpods = calculate_cojpods(n,pods,jpod);
        window_results.CoJPoDs = cojpods;
    end

end

function cojpods = calculate_cojpods(n,pods,jpod)

    jpods = ones(n,1) .* jpod;
    cojpods = (jpods ./ pods).';
    
end

function [dide,si,sv] = calculate_dide(n,pods,g,g_refs,p)

    dide = eye(n);
    
    for i = 1:n
        for j = 1:n
            if (isnan(pods(j)))
                dide(i,j) = NaN;
            elseif (i ~= j)
                dide(i,j) = p((g_refs == 2) & (g(:,i) == 1) & (g(:,j) == 1),:) / pods(j);
            end
        end
    end

    dide_pods = ((dide - eye(n)) .* repmat(pods,1,n));
    si = sum(dide_pods,2);
    sv = sum(dide_pods,1).';
    
end

function [jpod,fsi,pce] = calculate_indicators(n,pods,g_refs,p)

    jpod = p(g_refs == n,:);
    fsi = min(max(sum(pods,'omitnan') / (1 - p(g_refs == 0,:)),1),n);
    pce = sum(p(g_refs >= 2,:)) / sum(p(g_refs >= 1,:));
    
end

%% PLOTTING

function plot_indicators(ds,id)

    jpod = smooth_data(ds.Indicators(:,1));
    fsi = smooth_data(ds.Indicators(:,2));
    pce = smooth_data(ds.Indicators(:,3));
    
    if (ds.ByGroups)
        y_limits_fsi = [1 ds.Groups];
    else
        y_limits_fsi = [1 ds.N];
    end  

    f = figure('Name','Cross-Entropy Measures > Indicators','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);
    
    sub_1 = subplot(2,2,1:2);
    plot(sub_1,ds.DatesNum,jpod);
    set(sub_1,'YLim',plot_limits(jpod,0.1,0));
    t1 = title(sub_1,'Joint Probability of Default');
    set(t1,'Units','normalized');
    t1_position = get(t1,'Position');
    set(t1,'Position',[0.4783 t1_position(2) t1_position(3)]);
    
    sub_2 = subplot(2,2,3);
    plot(sub_2,ds.DatesNum,fsi);
    set(sub_2,'YLim',y_limits_fsi,'YTick',y_limits_fsi(1):y_limits_fsi(2));
    title(sub_2,'Financial Stability Index');
    
    sub_3 = subplot(2,2,4);
    plot(sub_3,ds.DatesNum,pce);
    set(sub_3,'YLim',plot_limits(pce,0,0));
    title(sub_3,'Probability of Cascade Effects');
    
    set([sub_1 sub_2 sub_3],'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
    set([sub_1 sub_2 sub_3],'XGrid','on','YGrid','on');
    
    if (ds.MonthlyTicks)
        date_ticks([sub_1 sub_2 sub_3],'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        date_ticks([sub_1 sub_2 sub_3],'x','yyyy','KeepLimits');
    end

    figure_title('Indicators');

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_dide(ds,id)

    if (ds.ByGroups)
        n = ds.Groups;
        labels = ds.GroupShortNames.';
    else
        n = ds.N;
        labels = ds.FirmNames;
    end

    dide = ds.AverageDiDe;
    didev = dide(:);
    
    [dide_x,dide_y] = meshgrid(1:n,1:n);
    dide_x = dide_x(:) + 0.5;
    dide_y = dide_y(:) + 0.5;
    
    dide_txt = cellstr(num2str(didev,'~%.4f'));

    for i = 1:n^2
        didev_i = didev(i);

        if (didev_i == 0)
            dide_txt{i} = '0';
        elseif (didev_i == 1)
            dide_txt{i} = '';
        end
    end
    
    lt_indices = (dide < 0.2) & (dide ~= 1);
    ge_indices = (dide >= 0.2) & (dide ~= 1);
    
    dide(lt_indices) =  0;
    dide(ge_indices) =  1;
    dide = dide - (eye(n) .* 0.5);
    dide = padarray(dide,[1 1],'post');
    
    didev = dide(:);
    didev_ones = any(didev == 1);
    didev_zeros = any(didev == 0);

    f = figure('Name','Cross-Entropy Measures > Average Distress Dependency','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    pcolor(dide);
    
    if (didev_ones && didev_zeros)
        colormap([1 1 1; 0.65 0.65 0.65; 0.749 0.862 0.933]);
    else
        if (didev_ones)
            colormap([0.65 0.65 0.65; 0.749 0.862 0.933]);
        else
            colormap([1 1 1; 0.65 0.65 0.65]);
        end
    end
        
    text(dide_x,dide_y,dide_txt,'HorizontalAlignment','center');
    axis image;

    ax = gca();
    set(ax,'TickLength',[0 0]);
    set(ax,'XAxisLocation','top','XTick',1.5:(n + 0.5),'XTickLabels',labels,'XTickLabelRotation',45);
    set(ax,'YDir','reverse','YTick',1.5:(n + 0.5),'YTickLabels',labels,'YTickLabelRotation',45);
    
    figure_title('Average Distress Dependency');

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_sequence_dide(ds,id)

    if (ds.ByGroups)
        n = ds.Groups;
    else
        n = ds.N;
    end
    
    t = ds.T;
    dn = ds.DatesNum;
    mt = ds.MonthlyTicks;
    
    ts_si = ds.SI;
    ts_sv = ds.SV;

    data = [repmat({dn},1,n); mat2cell(ts_si,t,ones(1,n)); mat2cell(ts_sv,t,ones(1,n))];
    
    if (ds.ByGroups)
        sequence_titles = ds.GroupShortNames.';
    else
        sequence_titles = ds.FirmNames;
    end

    x_limits = [dn(1) dn(end)];

    y_limits = [0 1];
    y_ticks = 0:0.1:1;

    core = struct();

    core.N = n;
    core.Data = data;
    core.Function = @(subs,data)plot_function(subs,data);

    core.OuterTitle = 'Cross-Entropy Measures > Distress Dependency Indicators Time Series';
    core.InnerTitle = 'Distress Dependency Indicators Time Series';
    core.SequenceTitles = sequence_titles;

    core.PlotsAllocation = [2 1];
    core.PlotsSpan = {1 2};
    core.PlotsTitle = [repmat({'Systemic Importance'},1,n); repmat({'Systemic Vulnerability'},1,n)];

    core.XDates = {mt mt};
    core.XGrid = {true true};
    core.XLabel = {[] []};
    core.XLimits = {x_limits x_limits};
    core.XRotation = {[] []};
    core.XTick = {[] []};
    core.XTickLabels = {[] []};

    core.YGrid = {true true};
    core.YLabel = {[] []};
    core.YLimits = {y_limits y_limits};
    core.YRotation = {[] []};
    core.YTick = {y_ticks y_ticks};
    core.YTickLabels = {[] []};

    sequential_plot(core,id);
    
    function plot_function(subs,data)
        
        x = data{1};
        si = data{2};
        sv = data{3};
        
        d = min(find(isnan(si),1,'first'),find(isnan(sv),1,'first'));
        
        if (isempty(d))
            xd = [];
        else
            xd = x(d) - 1;
        end

        plot(subs(1),x,si,'Color',[0.000 0.447 0.741]);

        if (~isempty(xd))
            hold(subs(1),'on');
                plot(subs(1),[xd xd],get(subs(1),'YLim'),'Color',[1 0.4 0.4]);
            hold(subs(1),'off');
        end
    
        plot(subs(2),x,sv,'Color',[0.000 0.447 0.741]);

        if (~isempty(xd))
            hold(subs(2),'on');
                plot(subs(2),[xd xd],get(subs(2),'YLim'),'Color',[1 0.4 0.4]);
            hold(subs(2),'off');
        end
        
    end

end

function plot_sequence_cojpods(ds,id)

    if (ds.ByGroups)
        n = ds.Groups;
    else
        n = ds.N;
    end

    t = ds.T;
    dn = ds.DatesNum;
    mt = ds.MonthlyTicks;
    
    ts = smooth_data(ds.CoJPoDs);

    data = [repmat({dn},1,n); mat2cell(ts,t,ones(1,n))];

    if (ds.ByGroups)
        sequence_titles = ds.GroupShortNames.';
    else
        sequence_titles = ds.FirmNames;
    end
    
    plots_title = repmat({' '},1,n);
    
    x_limits = [dn(1) dn(end)];
	y_limits = plot_limits(ts,0.1,0);
    
    y_tick_labels = @(x)sprintf('%.2f%%',x .* 100);

    core = struct();

    core.N = n;
    core.Data = data;
    core.Function = @(subs,data)plot_function(subs,data);

    core.OuterTitle = 'Cross-Entropy Measures > CoJPoDs Time Series';
    core.InnerTitle = 'CoJPoDs Time Series';
    core.SequenceTitles = sequence_titles;

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
    core.YTickLabels = {y_tick_labels};

    sequential_plot(core,id);
    
    function plot_function(subs,data)

        x = data{1};
        y = data{2};
        
        d = find(isnan(y),1,'first');
        
        if (isempty(d))
            xd = [];
        else
            xd = x(d) - 1;
        end
        
        plot(subs(1),x,y,'Color',[0.000 0.447 0.741]);

        if (~isempty(xd))
            hold(subs(1),'on');
                plot(subs(1),[xd xd],get(subs(1),'YLim'),'Color',[1 0.4 0.4]);
            hold(subs(1),'off');
        end

    end

end
