% [INPUT]
% ds = A structure representing the dataset.
% sn = A string representing the serial number of the result file.
% temp = A string representing the full path to the Excel spreadsheet used as template for the result file.
% out = A string representing the full path to the Excel spreadsheet to which the results are written, eventually replacing the previous ones.
% cvm = A string representing the method for calculating the critical values (optional, default='WB'):
%   - 'FS' for finite sample critical values;
%   - 'WB' for wild bootstrap.
% cvq = A float [0.90,0.99] representing the quantile of the critical values (optional, default=0.95).
% lag_max = An integer [0,10] representing the maximum lag order to be evaluated for the Augmented Dickey-Fuller test (optional, default=0).
% lag_sel = A string representing the lag order selection criteria for the Augmented Dickey-Fuller test (optional, default='FIX'):
%   - 'AIC' for Akaike's Information Criterion;
%   - 'BIC' for Bayesian Information Criterion;
%   - 'FIX' to use a fixed lag order;
%   - 'FPE' for Final Prediction Error;
%   - 'HQIC' for Hannan-Quinn Information Criterion.
% mbd = An integer [3,Inf) representing the minimum duration of a bubble in days (optional, default=NaN).
%   If NaN is provided, then an optimal value based on the number of observations is used.
% analyze = A boolean that indicates whether to analyse the results and display plots (optional, default=false).
%
% [OUTPUT]
% result = A structure representing the original dataset inclusive of intermediate and final calculations.
% stopped = A boolean that indicates whether the process has been stopped through user input.

function [result,stopped] = run_bubbles_detection(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('ds',@(x)validateattributes(x,{'struct'},{'nonempty'}));
        ip.addRequired('sn',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addRequired('temp',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addRequired('out',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addOptional('cvm','WB',@(x)any(validatestring(x,{'FS' 'WB'})));
        ip.addOptional('cvq',0.95,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.90 '<=' 0.99 'scalar'}));
        ip.addOptional('lag_max',0,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 0 '<=' 10 'scalar'}));
        ip.addOptional('lag_sel','FIX',@(x)any(validatestring(x,{'AIC' 'BIC' 'FIX' 'FPE' 'HQIC'})));
        ip.addOptional('mbd',NaN,@(x)validateattributes(x,{'double'},{'real' 'scalar'}));
        ip.addOptional('analyze',false,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    ds = validate_dataset(ipr.ds,'BubblesDetection');
    sn = ipr.sn;
    temp = validate_template(ipr.temp);
    out = validate_output(ipr.out);
    cvm = ipr.cvm;
    cvq = ipr.cvq;
    lag_max = ipr.lag_max;
    lag_sel = ipr.lag_sel;
    mbd = validate_mbd(ipr.mbd,ds.T);
    analyze = ipr.analyze;

    nargoutchk(1,2);

    [result,stopped] = run_bubbles_detection_internal(ds,sn,temp,out,cvm,cvq,lag_max,lag_sel,mbd,analyze);

end

function [result,stopped] = run_bubbles_detection_internal(ds,sn,temp,out,cvm,cvq,lag_max,lag_sel,mbd,analyze)

    result = [];
    stopped = false;
    e = [];

    ds = initialize(ds,sn,cvm,cvq,lag_max,lag_sel,mbd);
    n = ds.N;
    t = ds.T;

    bar = waitbar(0,'Initializing bubbles detection measures...','CreateCancelBtn',@(src,event)setappdata(gcbf(),'Stop', true));
    setappdata(bar,'Stop',false);
    cleanup = onCleanup(@()delete(bar));

    pause(1);
    waitbar(0,bar,'Calculating bubbles detection measures...');
    pause(1);

    try

        p = ds.Prices;

        for i = 1:n
            waitbar((i - 1) / n,bar,['Calculating bubbles detection measures for ' ds.FirmNames{i} '...']);

            if (getappdata(bar,'Stop'))
                stopped = true;
                break;
            end

            offset = min(ds.Defaults(i) - 1,t);
            p_i = p(1:offset,i);

            [bsadfs,cvs,detection,breakdown] = psy_bubbles_detection(p_i,ds.CVM,ds.CVQ,ds.LagMax,ds.LagSel,ds.MBD);
            ds.BSADFS(1:offset,i) = bsadfs;
            ds.CVS(1:offset,i) = cvs;
            ds.BUB(1:offset,i) = detection(:,1);
            ds.BMPH(1:offset,i) = detection(:,2);
            ds.BRPH(1:offset,i) = detection(:,3);
            ds.Breakdowns{i} = breakdown;

            if (getappdata(bar,'Stop'))
                stopped = true;
                break;
            end

            waitbar(i / n,bar);
        end

    catch e
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
    waitbar(1,bar,'Finalizing cross-sectional measures...');
    pause(1);

    try
        ds = finalize(ds);
    catch e
        delete(bar);
        rethrow(e);
    end

    pause(1);
    waitbar(1,bar,'Writing bubbles detection measures...');
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

function ds = initialize(ds,sn,cvm,cvq,lag_max,lag_sel,mbd)

    n = ds.N;
    t = ds.T;

    ds.Result = 'BubblesDetection';
    ds.ResultDate = now(); %#ok<TNOW1> 
    ds.ResultAnalysis = @(ds)analyze_result(ds);
    ds.ResultSerial = sn;

    ds.CVM = cvm;
    ds.CVQ = cvq;
    ds.LagMax = lag_max;
    ds.LagSel = lag_sel;
    ds.MBD = mbd;

    if (strcmp(ds.LagSel,'FIX'))
        label = [' (CVM= ' ds.CVM ', CVQ=' num2str(ds.CVQ * 100) '%, L=' num2str(ds.LagMax) ')'];
    else
        label = [' (CVM= ' ds.CVM ', CVQ=' num2str(ds.CVQ * 100) '%, LM=' num2str(ds.LagMax) ', LS=' ds.LagSel ')'];
    end

    ds.LabelsMeasuresSimple = {'BUB' 'BMPH' 'BRPH' 'Breakdowns'};
    ds.LabelsMeasures = [strcat(ds.LabelsMeasuresSimple(1:3),{label}) ds.LabelsMeasuresSimple(4)];

    ds.LabelsIndicatorsSimple = {'BC' 'BCP'};
    ds.LabelsIndicators = {['BC' label] ['BCP' label]};

    ds.LabelsSheetsSimple = [ds.LabelsMeasuresSimple {'Indicators'}];
    ds.LabelsSheets = [ds.LabelsMeasures {'Indicators'}];

    ds.BSADFS = NaN(t,n);
    ds.CVS = NaN(t,n);
    ds.BUB = NaN(t,n);
    ds.BMPH = NaN(t,n);
    ds.BRPH = NaN(t,n);
    ds.Breakdowns = cell(1,n);

    ds.Indicators = NaN(t,numel(ds.LabelsIndicators));

    ds.ComparisonReferences = {'Indicators' 1 strcat({'BD-'},ds.LabelsIndicatorsSimple{1})};

end

function ds = finalize(ds)

    bc = sum(ds.BUB .* ds.Capitalizations,2,'omitnan');
    bcp = bc ./ sum(ds.Capitalizations,2,'omitnan');

    ds.Indicators = [bc bcp];

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

    lc = numel(ds.LabelsSheetsSimple);

    dates_str = cell2table(ds.DatesStr,'VariableNames',{'Date'});

    for i = 1:(lc - 2)
        sheet = ds.LabelsSheetsSimple{i};
        measure = strrep(sheet,' ','');

        tab = [dates_str array2table(ds.(measure),'VariableNames',ds.FirmNames)];
        writetable(tab,out,'FileType','spreadsheet','Sheet',sheet,'WriteRowNames',true);
    end

    breakdowns_count = cellfun(@(x)size(x,1),ds.Breakdowns);
    breakdowns_firms = arrayfun(@(x)repelem(ds.FirmNames(x),breakdowns_count(x),1),1:ds.N,'UniformOutput',false);
    breakdowns_firms = vertcat(breakdowns_firms{:});
    breakdowns_firms = cell2table(breakdowns_firms,'VariableNames',{'Firm'});

    tab = [breakdowns_firms array2table(cell2mat(ds.Breakdowns.'),'VariableNames',{'Start' 'Peak' 'End' 'Duration', 'Boom Phase' 'Burst Phase'})];
    writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{lc - 1},'WriteRowNames',true);

    tab = [dates_str array2table(ds.Indicators,'VariableNames',ds.LabelsIndicatorsSimple)];
    writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{lc},'WriteRowNames',true);

    worksheets_batch(out,ds.LabelsSheetsSimple,ds.LabelsSheets);

end

%% PLOTTING

function analyze_result(ds)

    safe_plot(@(id)plot_sequence(ds,id));
    safe_plot(@(id)plot_indicators(ds,id));

end

function plot_indicators(ds,id)

    bc = ds.Indicators(:,1);
    bcp = ds.Indicators(:,2);

    y_limits_bc = plot_limits(bc,0.1,0);
    y_limits_bcp = [0 100];

    f = figure('Name','Bubbles Detection Measures > Indicators','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    sub_1 = subplot(2,1,1);
    plot(sub_1,ds.DatesNum,smooth_data(bc),'Color',[0.000 0.447 0.741]);
    set(sub_1,'YLim',y_limits_bc);
    t1 = title(sub_1,ds.LabelsIndicators{1});
    set(t1,'Units','normalized');
    t1_position = get(t1,'Position');
    set(t1,'Position',[0.4783 t1_position(2) t1_position(3)]);

    sub_2 = subplot(2,1,2);
    area(sub_2,ds.DatesNum,smooth_data(bcp),'EdgeColor',[0.000 0.447 0.741],'FaceAlpha',0.5,'FaceColor',[0.749 0.862 0.933]);
    set(sub_2,'YLim',y_limits_bcp);
    t2 = title(sub_2,ds.LabelsIndicators{2});
    set(t2,'Units','normalized');
    t2_position = get(t2,'Position');
    set(t2,'Position',[0.4783 t2_position(2) t2_position(3)]);

    set([sub_1 sub_2],'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
    set([sub_1 sub_2],'XGrid','on','YGrid','on');

    if (ds.MonthlyTicks)
        date_ticks([sub_1 sub_2],'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        date_ticks([sub_1 sub_2],'x','yyyy','KeepLimits');
    end

    figure_title(f,'Indicators');

    maximize_figure(f);

end

function plot_sequence(ds,id)

    n = ds.N;
    t = ds.T;
    dn = ds.DatesNum;
    mt = ds.MonthlyTicks;

    bmph = ds.BMPH;
    brph = ds.BRPH;

    data_bsadfs = ds.BSADFS;
    data_cvs = ds.CVS;
    data_ps = smooth_data(ds.Prices);
    data_ps_bm = data_ps .* bmph;
    data_ps_br = data_ps .* brph;
    data_ps_ot = NaN(t,n);

    data_bds = cell(1,n);

    for i = 1:n
        offset = min(ds.Defaults(i) - 1,t);

        bsadfs_i = data_bsadfs(1:offset,i);
        idx = find(bsadfs_i ~= 0,1,'first');
        data_bsadfs(1:idx,i) = bsadfs_i(idx + 1);
        data_bsadfs(1:offset,i) = smooth_data(data_bsadfs(1:offset,i));

        cvs_i = data_cvs(1:offset,i);
        idx = find(isnan(cvs_i),1,'last');
        data_cvs(1:idx,i) = cvs_i(idx + 1);
        data_cvs(1:offset,i) = smooth_data(data_cvs(1:offset,i));

        data_ps_ot(1:offset,i) = data_ps(1:offset,i) .* ~(bmph(1:offset,i) | bmph(1:offset,i));

        bd_i = ds.Breakdowns{i};

        bm_count = sum(bd_i(:,5));
        br_count = sum(bd_i(:,6));
        ot_count = t - (bm_count + br_count);
        data_bds{i} = [ot_count bm_count br_count];
    end

    data_ones = ones(1,n);
    data = [repmat({dn},1,n); mat2cell(data_bsadfs,t,data_ones); mat2cell(data_cvs,t,data_ones); mat2cell(data_ps,t,data_ones); mat2cell(data_ps_bm,t,data_ones); mat2cell(data_ps_br,t,data_ones); mat2cell(data_ps_ot,t,data_ones); data_bds];

    plots_title = [repmat({'Bubbles'},1,n); repmat({'Model'},1,n); repmat({'Breakdown - Percentage'},1,n); repmat({'Breakdown - Count'},1,n)];

    x_limits = [dn(1) dn(end)];

    core = struct();

    core.N = n;
    core.Data = data;
    core.Function = @(subs,data)plot_function(subs,data);

    core.OuterTitle = 'Bubbles Detection Measures > Time Series';
    core.InnerTitle = 'Time Series';
    core.SequenceTitles = ds.FirmNames;

    core.PlotsAllocation = [7 4];
    core.PlotsSpan = {[1:3 5:7 9:11 13:15] [21:23 25:27] [4 8 12] [20 24 28]};
    core.PlotsTitle = plots_title;

    core.XDates = {mt mt [] []};
    core.XGrid = {true true false false};
    core.XLabel = {[] [] [] []};
    core.XLimits = {x_limits x_limits [] []};
    core.XRotation = {45 45 [] []};
    core.XTick = {[] [] [] []};
    core.XTickLabels = {[] [] [] []};

    core.YGrid = {true true false true};
    core.YLabel = {[] [] [] []};
    core.YLimits = {[] [] [] []};
    core.YRotation = {[] [] [] []};
    core.YTick = {[] [] [] []};
    core.YTickLabels = {[] [] [] []};

    sequential_plot(core,id);

    function plot_function(subs,data)

        x = data{1};
        bsadfs = data{2};
        cvs = data{3};
        ps = data{4};
        ps_bm = data{5};
        ps_br = data{6};
        ps_ot = data{7};
        bd = data{8};

        d = find(isnan(ps),1,'first');

        if (isempty(d))
            xd = [];
        else
            xd = x(d) - 1;
        end

        sub_1 = subs(1);
        a1 = area(sub_1,x,ps_ot,'EdgeColor','none','FaceAlpha',0.5,'FaceColor',[0.749 0.862 0.933]);
        hold(sub_1,'on');
            a2 = area(sub_1,x,ps_bm,'EdgeColor','none','FaceColor',[0.200 0.627, 0.173]);
            a3 = area(sub_1,x,ps_br,'EdgeColor','none','FaceColor',[0.984 0.502 0.447]);
            plot(sub_1,x,ps,'Color',[0.000 0.000 0.000])
        hold(sub_1,'off');

        if (~isempty(xd))
            hold(sub_1,'on');
                plot(sub_1,[xd xd],get(sub_1,'YLim'),'Color',[1.000 0.400 0.400]);
            hold(sub_1,'off');
        end

        l = legend(sub_1,[a1 a2 a3],'None','Boom Phase','Burst Phase','Location','south','Orientation','horizontal');
        set(l,'Units','normalized');
        l_position = get(l,'Position');
        set(l,'Position',[l_position(1) 0.3645 l_position(3) l_position(4)]);

        sub_2 = subs(2);
        p1 = plot(sub_2,x,bsadfs,'Color',[0.000 0.447 0.741]);
        hold(sub_2,'on');
            p2 = plot(sub_2,x,cvs,'Color',[0.000 0.000 0.000],'LineStyle','--');
        hold(sub_2,'off');

        if (~isempty(xd))
            hold(sub_2,'on');
                plot(sub_2,[xd xd],get(sub_2,'YLim'),'Color',[1.000 0.400 0.400]);
            hold(sub_2,'off');
        end

        legend(sub_2,[p1 p2],'BSADF','Critical Values','Location','northwest');

        sub_3 = subs(3);
        pc1 = pie(sub_3,bd);
        hpc1 = findobj(pc1,'Type','patch');
        set(hpc1(1),'FaceAlpha',0.5,'FaceColor',[0.749 0.862 0.933]);
        set(hpc1(2),'FaceColor',[0.200 0.627, 0.173]);
        set(hpc1(3),'FaceColor',[0.984 0.502 0.447]);

        sub_4 = subs(4);
        b1 = bar(sub_4,1,bd(1));
        set(b1,'FaceAlpha',0.5,'FaceColor',[0.749 0.862 0.933]);
        hold(sub_4,'on');
            b2 = bar(sub_4,2,bd(2));
            set(b2,'FaceColor',[0.200 0.627, 0.173]);
            b3 = bar(sub_4,3,bd(3));
            set(b3,'FaceColor',[0.984 0.502 0.447]);
        hold(sub_4,'off');

    end

end

%% VALIDATION

function mbd = validate_mbd(mbd,t)

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

function out = validate_output(out)

    [path,name,extension] = fileparts(out);

    if (~strcmpi(extension,'.xlsx'))
        out = fullfile(path,[name extension '.xlsx']);
    end

end

function temp = validate_template(temp)

    sheets = {'BUB' 'BMPH' 'BRPH' 'Breakdowns' 'Indicators'};
    file_sheets = validate_xls(temp,'T');

    if (~all(ismember(sheets,file_sheets)))
        error(['The template must contain the following sheets: ' sheets{1} sprintf(', %s',sheets{2:end}) '.']);
    end

    worksheets_batch(temp,sheets);

end
