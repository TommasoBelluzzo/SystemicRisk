% [INPUT]
% ds = A structure representing the dataset.
% sn = A string representing the serial number of the result file.
% temp = A string representing the full path to the Excel spreadsheet used as template for the result file.
% out = A string representing the full path to the Excel spreadsheet to which the results are written, eventually replacing the previous ones.
% bw = An integer [21,252] representing the dimension of each rolling window (optional, default=252).
% f = A float [0.05,0.20] representing the percentage of observations to be included in tails used to calculate Average Chi and Asymptotic Dependency Rate (optional, default=0.10).
% pt = A float [0,1) representing the initial penantly term for underrepresented samples with respect to the bandwidth used to calculate Average Chi and Asymptotic Dependency Rate (optional, default=0.5).
% a = A float [0.01,0.10] representing the target quantile used to calculate the Financial Risk Meter (optional, default=0.05).
% ms = An integer [50,1000] representing the maximum number of steps used to calculate the Financial Risk Meter (optional, default=100).
% analyze = A boolean that indicates whether to analyse the results and display plots (optional, default=false).
%
% [OUTPUT]
% result = A structure representing the original dataset inclusive of intermediate and final calculations.
% stopped = A boolean that indicates whether the process has been stopped through user input.

function [result,stopped] = run_tail_dependence(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('ds',@(x)validateattributes(x,{'struct'},{'nonempty'}));
        ip.addRequired('sn',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addRequired('temp',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addRequired('out',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addOptional('bw',252,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 21 '<=' 252 'scalar'}));
        ip.addOptional('f',0.10,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.05 '<=' 0.2 'scalar'}));
        ip.addOptional('pt',0.5,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0 '<' 1 'scalar'}));
        ip.addOptional('a',0.05,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.01 '<=' 0.10 'scalar'}));
        ip.addOptional('ms',100,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 50 '<=' 1000 'scalar'}));
        ip.addOptional('analyze',false,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    ds = validate_dataset(ipr.ds,'TailDependence');
    sn = ipr.sn;
    temp = validate_template(ipr.temp);
    out = validate_output(ipr.out);
    bw = ipr.bw;
    f = ipr.f;
    pt = ipr.pt;
    a = ipr.a;
    ms = ipr.ms;
    analyze = ipr.analyze;

    nargoutchk(1,2);

    [result,stopped] = run_tail_dependence_internal(ds,sn,temp,out,bw,f,pt,a,ms,analyze);

end

function [result,stopped] = run_tail_dependence_internal(ds,sn,temp,out,bw,f,pt,a,ms,analyze)

    result = [];
    stopped = false;
    e = [];

    ds = initialize(ds,sn,bw,f,pt,a,ms);
    n = ds.N;
    t = ds.T;

    step_1 = 0.1;
    step_2 = 1 - step_1;

    rng(double(bitxor(uint16('T'),uint16('B'))));
    cleanup_1 = onCleanup(@()rng('default'));

    bar = waitbar(0,'Initializing tail dependence measures...','CreateCancelBtn',@(src,event)setappdata(gcbf(),'Stop',true));
    setappdata(bar,'Stop',false);
    cleanup_2 = onCleanup(@()delete(bar));

    pause(1);
    waitbar(0,bar,'Calculating tail dependence measures (step 1 of 2)...');
    pause(1);

    try

        rf = ds.Returns;

        [chi,~] = asymptotic_tail_dependence(rf,'L',ds.BW,ds.F,ds.PT);
        [achi,adr] = chi_metrics(chi);

        ds.Indicators(:,1) = achi;
        ds.Indicators(:,2) = adr;

    catch e
    end

    if (~isempty(e))
        delete(bar);
        rethrow(e);
    end

    pause(1);
    waitbar(step_1,bar,'Calculating tail dependence measures (step 2 of 2)...');
    pause(1);

    try

        rf = ds.Returns;
        sv = ds.StateVariables;
        vars = [rf sv];

        futures(1:n) = parallel.FevalFuture;
        futures_max = 0;
        futures_results = cell(t,1);

        for i = 1:n
            offset = min(ds.Defaults(i) - 1,t);
            y = vars(1:offset,i);
            x = vars(1:offset,[1:i-1 i+1:end]);
            futures(i) = parfeval(@main_loop,1,t,y,x,ds.BW,ds.A,ds.MS);
        end

        for i = 1:n
            if (getappdata(bar,'Stop'))
                stopped = true;
                break;
            end

            [future_index,value] = fetchNext(futures);
            futures_results{future_index} = value;

            futures_max = max([future_index futures_max]);
            waitbar(step_1 + (step_2 * ((futures_max - 1) / n)),bar);

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
    waitbar(1,bar,'Finalizing tail dependence measures (step 2 of 2)...');
    pause(1);

    try
        ds = finalize(ds,futures_results);
    catch e
        delete(bar);
        rethrow(e);
    end

    pause(1);
    waitbar(1,bar,'Writing tail dependence measures...');
    pause(1);

    try
        write_results(ds,temp,out);
        delete(bar);
    catch e
        delete(bar);
        rethrow(e);
    end

    if (analyze)
        analyze_result(ds);
    end

    result = ds;

end

%% PROCESS

function ds = initialize(ds,sn,bw,f,pt,a,ms)

    n = ds.N;
    t = ds.T;

    ds.Result = 'TailDependence';
    ds.ResultDate = now();
    ds.ResultAnalysis = @(ds)analyze_result(ds);
    ds.ResultSerial = sn;

    ds.A = a;
    ds.BW = bw;
    ds.F = f;
    ds.MS = ms;
    ds.PT = pt;

    am_label = [' (F=' num2str(ds.F * 100) '%, PT=' num2str(ds.PT * 100) '%)'];
    frm_label = [' (A=' num2str(ds.A * 100) '%, MS=' num2str(ds.MS) ')'];

    ds.LabelsMeasuresSimple = {'FRM Lambdas'};
    ds.LabelsMeasures = {['FRM Lambdas' frm_label]};

    ds.LabelsIndicatorsSimple = {'ACHI' 'ADR' 'FRM'};
    ds.LabelsIndicators = {['ACHI' am_label] ['ADR' am_label] ['FRM' frm_label]};

    ds.LabelsSheetsSimple = [ds.LabelsMeasuresSimple {'Indicators'}];
    ds.LabelsSheets = [ds.LabelsMeasures {'Indicators'}];

    ds.FRMLambdas = NaN(t,n);

    ds.Indicators = NaN(t,numel(ds.LabelsIndicators));

    ds.ComparisonReferences = {'Indicators' [] strcat({'TD-'},strrep(ds.LabelsIndicatorsSimple,' ',''))};

end

function window_results = main_loop(t,y,x,bw,a,ms)

    windows_y = extract_rolling_windows(y,bw);
    windows_x = extract_rolling_windows(x,bw);

    lambda = NaN(t,1);

    for i = 1:numel(y)
        y_i = windows_y{i};

        x_i = windows_x{i};
        x_i(isnan(x_i)) = 0;

        [~,lambda_t] = lasso_quantile_regression(y_i,x_i,a,ms);
        lambda(i) = abs(lambda_t);
    end

    window_results = struct();
    window_results.Lambda = lambda;

end

function ds = finalize(ds,results)

    n = ds.N;

    for i = 1:n
        result = results{i};
        ds.FRMLambdas(:,i) = result.Lambda;
    end

    ds.Indicators(:,3) = mean(ds.FRMLambdas,2,'omitnan');

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
        measure = strrep(sheet,' ','');

        tab = [dates_str array2table(ds.(measure),'VariableNames',ds.FirmNames)];
        writetable(tab,out,'FileType','spreadsheet','Sheet',sheet,'WriteRowNames',true);
    end

    tab = [dates_str array2table(ds.Indicators,'VariableNames',strrep(ds.LabelsIndicatorsSimple,' ','_'))];
    writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{end},'WriteRowNames',true);

    worksheets_batch(out,ds.LabelsSheetsSimple,ds.LabelsSheets);

end

%% PLOTTING

function analyze_result(ds)

    safe_plot(@(id)plot_asymptotic_indicators(ds,id));
    safe_plot(@(id)plot_frm_lambdas(ds,id));
    safe_plot(@(id)plot_frm(ds,id));

end

function plot_asymptotic_indicators(ds,id)

    achi = smooth_data(ds.Indicators(:,1));
    adr = smooth_data(ds.Indicators(:,2));

    f = figure('Name','Tail Dependence Measures > Asymptotic Indicators','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    sub_1 = subplot(1,2,1);
    plot(sub_1,ds.DatesNum,achi,'Color',[0.000 0.447 0.741]);
    set(sub_1,'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
    set(sub_1,'XGrid','on','YGrid','on');
    title(sub_1,ds.LabelsIndicators(1));

    sub_2 = subplot(1,2,2);
    plot(sub_2,ds.DatesNum,adr,'Color',[0.000 0.447 0.741]);
    set(sub_2,'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
    set(sub_2,'XGrid','on','YGrid','on');
    title(sub_2,ds.LabelsIndicators(2));
    
    set([sub_1 sub_2],'YLim',[0 1]);
    
    if (ds.MonthlyTicks)
        date_ticks([sub_1 sub_2],'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        date_ticks([sub_1 sub_2],'x','yyyy','KeepLimits');
    end

    figure_title('Asymptotic Dependence Indicators');

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_frm_lambdas(ds,id)

    n = ds.N;
    t = ds.T;
    dn = ds.DatesNum;
    mt = ds.MonthlyTicks;

    ts = smooth_data(ds.FRMLambdas);

    data = [repmat({dn},1,n); mat2cell(ts,t,ones(1,n))];

    plots_title = repmat(ds.LabelsMeasures(1),1,n);

    x_limits = [dn(1) dn(end)];
    y_limits = plot_limits(ts,0.1);

    core = struct();

    core.N = n;
    core.Data = data;
    core.Function = @(subs,data)plot_function(subs,data);

    core.OuterTitle = 'Tail Dependence Measures > FRM Lambdas';
    core.InnerTitle = 'FRM Lambdas Time Series';
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

function plot_frm(ds,id)

    y = smooth_data(ds.Indicators(:,3));

    f = figure('Name','Tail Dependence Measures > Financial Risk Meter','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    sub_1 = subplot(1,6,1:5);
    plot(sub_1,ds.DatesNum,y,'Color',[0.000 0.447 0.741]);
    set(sub_1,'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
    set(sub_1,'XGrid','on','YGrid','on');
    title(sub_1,ds.LabelsIndicators(3));

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

    figure_title('Financial Risk Meter');

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

%% VALIDATION

function out_file = validate_output(out_file)

    [path,name,extension] = fileparts(out_file);

    if (~strcmpi(extension,'.xlsx'))
        out_file = fullfile(path,[name extension '.xlsx']);
    end

end

function temp = validate_template(temp)

    sheets = {'FRM Lambdas' 'Indicators'};
    file_sheets = validate_xls(temp,'T');

    if (~all(ismember(sheets,file_sheets)))
        error(['The template must contain the following sheets: ' sheets{1} sprintf(', %s',sheets{2:end}) '.']);
    end

    worksheets_batch(temp,sheets);

end
