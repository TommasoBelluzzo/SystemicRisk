% [INPUT]
% ds = A structure representing the dataset.
% sn = A string representing the serial number of the result file.
% temp = A string representing the full path to the Excel spreadsheet used as template for the result file.
% out = A string representing the full path to the Excel spreadsheet to which the results are written, eventually replacing the previous ones.
% bw = An integer [21,252] representing the dimension of each rolling window (optional, default=252).
% op = A string representing the option pricing model (optional, default='BSM'):
%   - 'BSM' for Black-Scholes-Merton.
%   - 'GC' for Gram-Charlier.
% lst = A float or a vector of floats (0,Inf) representing the long-term to short-term liabilities ratio(s) used to calculate D2C and D2D (optional, default=3).
% car = A float [0.03,0.20] representing the capital adequacy ratio used to calculate the D2C (optional, default=0.08).
% f = An integer [2,n], where n is the number of firms, representing the number of systematic risk factors used to calculate the DIP (optional, default=2).
% lgd = A float (0,1] representing the loss given default, complement to recovery rate, used to calculate the DIP (optional, default=0.55).
% l = A float [0.05,0.20] representing the importance sampling threshold used to calculate the DIP (optional, default=0.10).
% c = An integer [50,1000] representing the number of simulated samples used to calculate the DIP (optional, default=100).
% it = An integer [5,100] representing the number of iterations to perform to calculate the DIP (optional, default=5).
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
        ip.addRequired('sn',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addRequired('temp',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addRequired('out',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addOptional('bw',252,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 21 '<=' 252 'scalar'}));
        ip.addOptional('op','BSM',@(x)any(validatestring(x,{'BSM' 'GC'})));
        ip.addOptional('lst',3,@(x)validateattributes(x,{'double'},{'real' 'finite' '>' 0 'vector'}));
        ip.addOptional('car',0.08,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.03 '<=' 0.20 'scalar'}));
        ip.addOptional('f',2,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 2 'scalar'}));
        ip.addOptional('lgd',0.55,@(x)validateattributes(x,{'double'},{'real' 'finite' '>' 0 '<=' 1 'scalar'}));
        ip.addOptional('l',0.10,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.05 '<=' 0.20 'scalar'}));
        ip.addOptional('c',100,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 50 '<=' 1000 'scalar'}));
        ip.addOptional('it',5,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 5 '<=' 100 'scalar'}));
        ip.addOptional('k',0.95,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.90 '<=' 0.99 'scalar'}));
        ip.addOptional('analyze',false,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    ds = validate_dataset(ipr.ds,'Default');
    sn = ipr.sn;
    temp = validate_template(ipr.temp);
    out = validate_output(ipr.out);
    bw = ipr.bw;
    op = ipr.op;
    lst = validate_lst(ipr.lst,ds.N);
    car = ipr.car;
    f = validate_f(ipr.f,ds.N);
    lgd = ipr.lgd;
    l = ipr.l;
    c = ipr.c;
    it = ipr.it;
    k = ipr.k;
    analyze = ipr.analyze;

    nargoutchk(1,2);

    [result,stopped] = run_default_internal(ds,sn,temp,out,bw,op,lst,car,f,lgd,l,c,it,k,analyze);

end

function [result,stopped] = run_default_internal(ds,sn,temp,out,bw,op,lst,car,f,lgd,l,c,it,k,analyze)

    result = [];
    stopped = false;
    e = [];

    ds = initialize(ds,sn,bw,op,lst,car,f,lgd,l,c,it,k);
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
            futures(i) = parfeval(@main_loop_1,1,firms_data{i},offsets,r,ds.ST(i),ds.DT(i),ds.CAR,ds.OP);
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

        cl = ds.SCCACL;
        cl(isnan(cl)) = 0;
        windows_cl = extract_rolling_windows(cl,ds.BW);

        futures(1:t) = parallel.FevalFuture;
        futures_max = 0;
        futures_results = cell(t,1);

        for i = 1:t
            futures(i) = parfeval(@main_loop_2,1,windows_r{i},cds(i,:),lb(i,:),ds.F,ds.LGD,ds.L,ds.C,ds.IT,windows_cl{i},ds.K);
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
        analyze_result(ds);
    end

    result = ds;

end

%% PROCESS

function ds = initialize(ds,sn,bw,op,lst,car,f,lgd,l,c,it,k)

    q = [(0.900:0.025:0.975) 0.99];
    q = q(q >= k);

    n = ds.N;
    t = ds.T;

    ds.Result = 'Default';
    ds.ResultDate = now();
    ds.ResultAnalysis = @(ds)analyze_result(ds);
    ds.ResultSerial = sn;

    ds.A = 1 - k;
    ds.BW = bw;
    ds.C = c;
    ds.CAR = car;
    ds.DT = max(0.5,0.7 - (0.3 .* (1 ./ lst)));
    ds.F = f;
    ds.IT = it;
    ds.K = k;
    ds.L = l;
    ds.LGD = lgd;
    ds.LST = lst;
    ds.OP = op;
    ds.ST =  1 ./ (1 + lst);

    op_label =  [' (' ds.OP ')'];
    d2c_label =  [' (' ds.OP ', CAR=' num2str(ds.CAR * 100) '%)'];
    dip_label =  [' (LGD=' num2str(ds.LGD * 100) '%, F=' num2str(ds.F) ', L=' num2str(ds.L * 100) '%, IT=' num2str(it) ')'];
    scca_label = [' (' ds.OP ', K=' num2str(ds.K * 100) '%)'];

    ds.LabelsMeasuresSimple = {'D2D' 'D2C' 'SCCA EL' 'SCCA CL'};
    ds.LabelsMeasures = {['D2D' op_label] ['D2C' d2c_label] ['SCCA EL' op_label] ['SCCA CL' op_label]};

    ds.LabelsIndicatorsSimple = {'AD2D' 'AD2C' 'PD2D' 'PD2C' 'DIP' 'SCCA JES'};
    ds.LabelsIndicators = {['AD2D' op_label] ['AD2C' d2c_label] ['PD2D' op_label] ['PD2C' d2c_label] ['DIP' dip_label] ['SCCA JES' scca_label]};

    ds.LabelsSheetsSimple = [ds.LabelsMeasuresSimple {'Indicators'}];
    ds.LabelsSheets = [ds.LabelsMeasures {'Indicators'}];

    ds.D2D = NaN(t,n);
    ds.D2C = NaN(t,n);

    ds.SCCAAlphas = NaN(t,n);
    ds.SCCAEL = NaN(t,n);
    ds.SCCACL = NaN(t,n);
    ds.SCCAJVaRs = NaN(t,numel(q));

    ds.Indicators = NaN(t,numel(ds.LabelsIndicators));

    ds.ComparisonReferences = {'Indicators' [] strcat({'DE-'},strrep(ds.LabelsIndicatorsSimple,' ',''))};

end

function window_results = main_loop_1(firm_data,offsets,r,st,dt,car,op)

    t = size(firm_data,1);

    offset1 = min(offsets(1),t);
    cp_o = max(1e-6,firm_data(1:offset1,1));
    lb_o = max(1e-6,firm_data(1:offset1,3));
    db_o = (lb_o .* st) + (dt .* (lb_o .* (1 - st)));
    r_o = r(1:offset1);

    [va,vap] = kmv_structural(cp_o,db_o,r_o,1,op);
    [d2d,d2c] = default_metrics(va,vap(1),db_o,r_o,1,car);

    offset2 = min(min(offsets),t);
    cp_o = max(1e-6,firm_data(1:offset2,1));
    lb_o = max(1e-6,firm_data(1:offset2,3));
    db_o = (lb_o .* st) + (dt .* (lb_o .* (1 - st)));
    r_o = r(1:offset2);
    cds_o = firm_data(1:offset2,2);

    [va,vap] = kmv_structural(cp_o,db_o,r_o,1,op);
    [el,cl,a] = contingent_claims_analysis(va,vap,cds_o,db_o,r_o,1);

    window_results = struct();
    window_results.Offset1 = offset1;
    window_results.Offset2 = offset2;
    window_results.D2D = d2d;
    window_results.D2C = d2c;
    window_results.SCCAAlphas = a;
    window_results.SCCAEL = el;
    window_results.SCCACL = cl;

end

function window_results = main_loop_2(r,cds,lb,f,lgd,l,c,it,cl,k)

    window_results = struct();

    dip = distress_insurance_premium(r,cds,lb,f,lgd,l,c,it);
    window_results.DIP = dip;

    [jvars,jes] = mgev_joint_risk_metrics(cl,k);
    window_results.SCCAJVaRs = jvars;
    window_results.SCCAJES = jes;

end

function ds = finalize_1(ds,results)

    n = ds.N;
    t = ds.T;

    cp = ds.Capitalizations;
    lb = ds.Liabilities;
    r = max(0,ds.RiskFreeRate);

    for i = 1:n
        result = results{i};

        ds.D2D(1:result.Offset1,i) = result.D2D;
        ds.D2C(1:result.Offset1,i) = result.D2C;

        ds.SCCAAlphas(1:result.Offset2,i) = result.SCCAAlphas;
        ds.SCCAEL(1:result.Offset2,i) = result.SCCAEL;
        ds.SCCACL(1:result.Offset2,i) = result.SCCACL;
    end

    weights = cp ./ repmat(sum(cp,2,'omitnan'),1,n);
    d2d_avg = sum(ds.D2D .* weights,2,'omitnan');
    d2c_avg = sum(ds.D2C .* weights,2,'omitnan');

    cp = max(1e-6,sum(cp,2,'omitnan'));
    lbs = lb .* repmat(ds.ST,t,1);
    lbl = repmat(ds.DT,t,1) .* (lb .* (1 - repmat(ds.ST,t,1)));
    db = max(1e-6,sum(lbs + lbl,2,'omitnan'));

    [va,vap] = kmv_structural(cp,db,r,1,ds.OP);
    [d2d_por,d2c_por] = default_metrics(va,vap(1),db,r,1,ds.CAR);

    ds.Indicators(:,1) = d2d_avg;
    ds.Indicators(:,2) = d2c_avg;
    ds.Indicators(:,3) = d2d_por;
    ds.Indicators(:,4) = d2c_por;

    measures_len = numel(ds.LabelsMeasuresSimple);
    measures = cell(measures_len,1);

    for i = 1:measures_len
        measures{i} = ds.(strrep(ds.LabelsMeasuresSimple{i},' ',''));
    end

    [rc,rs] = kendall_rankings(measures);
    ds.RankingConcordance = rc;
    ds.RankingStability = rs;

end

function ds = finalize_2(ds,results)

    t = ds.T;

    for i = 1:t
        result = results{i};

        ds.SCCAJVaRs(i,:) = result.SCCAJVaRs;

        ds.Indicators(i,5) = result.DIP;
        ds.Indicators(i,6) = result.SCCAJES;
    end

    w = max(round(nthroot(ds.BW,1.81),0),5); 
    ds.Indicators(:,5) = sanitize_data(ds.Indicators(:,5),ds.DatesNum,w,[]);

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

    safe_plot(@(id)plot_distances(ds,id));
    safe_plot(@(id)plot_sequence(ds,'D2D',id));
    safe_plot(@(id)plot_sequence(ds,'D2C',id));
    safe_plot(@(id)plot_dip(ds,id));
    safe_plot(@(id)plot_scca(ds,id));
    safe_plot(@(id)plot_sequence(ds,'SCCA EL',id));
    safe_plot(@(id)plot_sequence(ds,'SCCA CL',id));
    safe_plot(@(id)plot_rankings(ds,id));

end

function plot_distances(ds,id)

    distances = ds.Indicators(:,1:4);

    op_label =  [' (' ds.OP ')'];
    d2c_label =  [' (' ds.OP ', CAR=' num2str(ds.CAR * 100) ')'];

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
    title(sub_1,['Average D2D' op_label]);

    sub_2 = subplot(2,2,2);
    plot(sub_2,ds.DatesNum,smooth_data(distances(:,3)),'Color',[0.000 0.447 0.741]);
    hold on;
        plot(sub_2,ds.DatesNum,zeros(ds.T,1),'Color',[1 0.4 0.4]);
    hold off;
    title(sub_2,['Average D2C' d2c_label]);

    sub_3 = subplot(2,2,3);
    plot(sub_3,ds.DatesNum,smooth_data(distances(:,2)),'Color',[0.000 0.447 0.741]);
    hold on;
        plot(sub_3,ds.DatesNum,zeros(ds.T,1),'Color',[1 0.4 0.4]);
    hold off;
    title(sub_3,['Portfolio D2D' op_label]);

    sub_4 = subplot(2,2,4);
    plot(sub_4,ds.DatesNum,smooth_data(distances(:,4)),'Color',[0.000 0.447 0.741]);
    hold on;
        plot(sub_4,ds.DatesNum,zeros(ds.T,1),'Color',[1 0.4 0.4]);
    hold off;
    title(sub_4,['Portfolio D2C' d2c_label]);

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
    title(sub_1,['DIP (LGD=' num2str(ds.LGD * 100) '%, F=' num2str(ds.F) ', L=' num2str(ds.L * 100) '%)']);

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

function plot_rankings(ds,id)

    labels = ds.LabelsMeasuresSimple;
    n = numel(labels);
    seq = 1:n;
    off = seq + 0.5;

    [rs,order] = sort(ds.RankingStability);
    rs_names = labels(order);

    rc = ds.RankingConcordance;
    rc(rc <= 0.5) = 0;
    rc(rc > 0.5) = 1;
    rc(logical(eye(n))) = 0.5;

    [rc_x,rc_y] = meshgrid(seq,seq);
    rc_x = rc_x(:) + 0.5;
    rc_y = rc_y(:) + 0.5;
    rc_text = cellstr(num2str(ds.RankingConcordance(:),'%.2f'));

    f = figure('Name','Default Measures > Rankings','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    sub_1 = subplot(1,2,1);
    bar(sub_1,seq,rs,'FaceColor',[0.749 0.862 0.933]);
    set(sub_1,'XTickLabel',rs_names,'XTickLabelRotation',45);
    set(sub_1,'YLim',[0 1]);
    title(sub_1,'Ranking Stability');

    sub_2 = subplot(1,2,2);
    pcolor(padarray(rc,[1 1],'post'));
    colormap([1 1 1; 0.65 0.65 0.65; 0.749 0.862 0.933]);
    axis('image');
    text(rc_x,rc_y,rc_text,'FontSize',9,'HorizontalAlignment','center');
    set(sub_2,'FontWeight','bold','TickLength',[0 0]);
    set(sub_2,'XAxisLocation','bottom','XTick',off,'XTickLabels',labels,'XTickLabelRotation',45);
    set(sub_2,'YDir','reverse','YTick',off,'YTickLabels',labels,'YTickLabelRotation',45);
    title(sub_2,'Ranking Concordance');

    figure_title('Rankings (Kendall''s W)');

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_scca(ds,id)

    el = sum(ds.SCCAEL,2,'omitnan');
    cl = sum(ds.SCCACL,2,'omitnan');
    alpha = cl ./ el;
    jes = ds.Indicators(:,6);

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
    title(sub_2,'Average Alpha');

    sub_3 = subplot(2,2,4);
    plot(sub_3,ds.DatesNum,smooth_data(jes),'Color',[0.000 0.447 0.741]);
    title(sub_3,['Joint ES (K=' sprintf('%.1f%%',(ds.K * 100)) ')']);

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

function plot_sequence(ds,target,id)

    is_distance = any(strcmp(target,{'D2C' 'D2D'}));

    n = ds.N;
    t = ds.T;
    dn = ds.DatesNum;
    mt = ds.MonthlyTicks;

    ts = smooth_data(ds.(strrep(target,' ','')));

    data = [repmat({dn},1,n); mat2cell(ts,t,ones(1,n))];

    [~,index] = ismember(target,ds.LabelsMeasuresSimple);
    plots_title = repmat(ds.LabelsMeasures(index),1,n);

    x_limits = [dn(1) dn(end)];

    if (is_distance)
        y_limits = plot_limits(ts,0.1,[],[],-1);
    else
        y_limits = plot_limits(ts,0.1);
    end

    core = struct();

    core.N = n;
    core.Data = data;
    core.Function = @(subs,data)plot_function(subs,data,is_distance);

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

    function plot_function(subs,data,is_distance)

        x = data{1};
        y = data{2};

        d = find(isnan(y),1,'first');

        if (isempty(d))
            xd = [];
        else
            xd = x(d) - 1;
        end

        plot(subs(1),x,y,'Color',[0.000 0.447 0.741]);

        if (is_distance)
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

%% VALIDATION

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

    if (~strcmpi(extension,'.xlsx'))
        out_file = fullfile(path,[name extension '.xlsx']);
    end

end

function f = validate_f(f,n)

    if (f > n)
        error(['The value of ''f'' is invalid. Expected input to be less than or equal to (' num2str(n) ').']);
    end

end

function temp = validate_template(temp)

    sheets = {'D2D' 'D2C' 'SCCA EL' 'SCCA CL' 'Indicators'};
    file_sheets = validate_xls(temp,'T');

    if (~all(ismember(sheets,file_sheets)))
        error(['The template must contain the following sheets: ' sheets{1} sprintf(', %s',sheets{2:end}) '.']);
    end

    worksheets_batch(temp,sheets);

end
