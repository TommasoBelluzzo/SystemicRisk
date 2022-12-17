% [INPUT]
% ds = A structure representing the dataset.
% sn = A string representing the serial number of the result file.
% temp = A string representing the full path to the Excel spreadsheet used as template for the result file.
% out = A string representing the full path to the Excel spreadsheet to which the results are written, eventually replacing the previous ones.
% bw = An integer [21,252] representing the dimension of each rolling window (optional, default=252).
% k = A float [0.90,0.99] representing the confidence level used to calculate the CATFIN (optional, default=0.99).
% g = A float [0.75,0.99] representing the weighting factor of the non-parametric value-at-risk used to calculate the CATFIN (optional, default=0.98).
% u = A float [0.01,0.10] representing the threshold of the GPD value-at-risk used to calculate the CATFIN (optional, default=0.05).
% f = A float [0.2,0.8] representing the percentage of components to include in the computation of the Absorption Ratio (optional, default=0.2).
% q = A float (0.5,1.0) representing the quantile threshold of Correlation Surprise and Turbulence Index (optional, default=0.75).
% analyze = A boolean that indicates whether to analyse the results and display plots (optional, default=false).
%
% [OUTPUT]
% result = A structure representing the original dataset inclusive of intermediate and final calculations.
% stopped = A boolean that indicates whether the process has been stopped through user input.

function [result,stopped] = run_component(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('ds',@(x)validateattributes(x,{'struct'},{'nonempty'}));
        ip.addRequired('sn',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addRequired('temp',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addRequired('out',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addOptional('bw',252,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 21 '<=' 252 'scalar'}));
        ip.addOptional('k',0.99,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.90 '<=' 0.99 'scalar'}));
        ip.addOptional('g',0.98,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.50 '<=' 0.99 'scalar'}));
        ip.addOptional('u',0.05,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.01 '<=' 0.10 'scalar'}));
        ip.addOptional('f',0.2,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.2 '<=' 0.8 'scalar'}));
        ip.addOptional('q',0.75,@(x)validateattributes(x,{'double'},{'real' 'finite' '>' 0.5 '<' 1 'scalar'}));
        ip.addOptional('analyze',false,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    ds = validate_dataset(ipr.ds,'Component');
    sn = ipr.sn;
    temp = validate_template(ipr.temp);
    out = validate_output(ipr.out);
    bw = ipr.bw;
    k = ipr.k;
    g = ipr.g;
    u = ipr.u;
    f = ipr.f;
    q = ipr.q;
    analyze = ipr.analyze;

    nargoutchk(1,2);

    [result,stopped] = run_component_internal(ds,sn,temp,out,bw,k,g,u,f,q,analyze);

end

function [result,stopped] = run_component_internal(ds,sn,temp,out,bw,k,g,u,f,q,analyze)

    result = [];
    stopped = false;
    e = [];

    ds = initialize(ds,sn,bw,k,g,u,f,q);
    t = ds.T;

    rng(double(bitxor(uint16('T'),uint16('B'))));
    cleanup_1 = onCleanup(@()rng('default'));

    bar = waitbar(0,'Initializing component measures...','CreateCancelBtn',@(src,event)setappdata(gcbf(),'Stop',true));
    setappdata(bar,'Stop',false);
    cleanup_2 = onCleanup(@()delete(bar));

    pause(1);
    waitbar(0,bar,'Calculating component measures...');
    pause(1);

    try

        windows_rf = extract_rolling_windows(ds.Returns,ds.BW);
        windows_rp = extract_rolling_windows(ds.CATFINReturns,ds.BW);

        futures(1:t) = parallel.FevalFuture;
        futures_max = 0;
        futures_results = cell(t,1);

        for i = 1:t
            futures(i) = parfeval(@main_loop,1,windows_rf{i},windows_rp{i},ds.A,ds.G,ds.U,ds.F);
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
    waitbar(1,bar,'Finalizing component measures...');
    pause(1);

    try
        ds = finalize(ds,futures_results);
    catch e
        delete(bar);
        rethrow(e);
    end

    pause(1);
    waitbar(1,bar,'Writing component measures...');
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

function ds = initialize(ds,sn,bw,k,g,u,f,q)

    n = ds.N;
    t = ds.T;

    r = ds.Returns;
    rw = 1 ./ (repmat(n,t,1) - sum(isnan(r),2));
    rp = sum(r .* repmat(rw,1,n),2,'omitnan');

    ds.Result = 'Component';
    ds.ResultDate = now(); %#ok<TNOW1> 
    ds.ResultAnalysis = @(ds)analyze_result(ds);
    ds.ResultSerial = sn;

    ds.A = 1 - k;
    ds.BW = bw;
    ds.G = g;
    ds.U = u;
    ds.F = f;
    ds.K = k;
    ds.Q = q;

    ds.CATFINReturns = rp;

    g_label = num2str(ds.G);
    u_label = num2str(ds.U);
    f_label = [num2str(ds.F * 100) '%'];
    k_label = [num2str(ds.K * 100) '%'];
    q_label = num2str(ds.Q);

    ds.LabelsCATFINVaRs = {['NP (G=' g_label ')'] ['GPD (U=' u_label ')'] 'GEV' 'SGED'};
    ds.LabelsPCAExplained = {'PC' 'EV'};

    ds.LabelsIndicatorsSimple = {'AR' 'CATFIN' 'CS' 'TI'};
    ds.LabelsIndicators = {['AR (F=' f_label ')'] ['CATFIN (K=' k_label ')'] ['CS (Q=' q_label ')'] ['TI (Q=' q_label ')']};

    ds.LabelsSheetsSimple = {'CATFIN VaRs' 'Indicators' 'PCA Overall Explained' 'PCA Overall Coefficients' 'PCA Overall Scores'};
    ds.LabelsSheets = {['CATFIN VaRs (K=' k_label ')'] 'Indicators' 'PCA Overall Explained' 'PCA Overall Coefficients' 'PCA Overall Scores'};

    ds.CATFINVaRs = NaN(t,4);
    ds.CATFINFirstCoefficients = NaN(1,4);
    ds.CATFINFirstExplained = NaN;

    ds.Indicators = NaN(t,numel(ds.LabelsIndicators));

    ds.PCACoefficients = cell(t,1);
    ds.PCAExplained = cell(t,1);
    ds.PCAExplainedSums = NaN(t,4);
    ds.PCAScores = cell(t,1);

    ds.PCACoefficientsOverall = NaN(n,n);
    ds.PCAExplainedOverall = NaN(n,1);
    ds.PCAExplainedSumsOverall = NaN(1,4);
    ds.PCAScoresOverall = NaN(t,n);

    ds.ComparisonReferences = {'Indicators' [] strcat({'CO-'},ds.LabelsIndicatorsSimple)};

end

function window_results = main_loop(rf,rp,a,g,u,f)

    window_results = struct();

    [var_np,var_gpd,var_gev,var_sged] = catfin(rp,a,g,u);
    window_results.CATFINVaRs = -1 .* [var_np var_gpd var_gev var_sged];

    [ar,cs,ti] = component_metrics(rf,f);
    window_results.AbsorptionRatio = ar;
    window_results.CorrelationSurprise = cs;
    window_results.TurbulenceIndex = ti;

    [coefficients,scores,explained] = pca_shorthand(rf,true);
    window_results.PCACoefficients = coefficients;
    window_results.PCAExplained = explained;
    window_results.PCAScores = scores;

end

function ds = finalize(ds,results)

    t = ds.T;

    for i = 1:t
        result = results{i};

        ds.CATFINVaRs(i,:) = result.CATFINVaRs;

        ds.Indicators(i,[1 3 4]) = [result.AbsorptionRatio result.CorrelationSurprise result.TurbulenceIndex];

        ds.PCACoefficients{i} = result.PCACoefficients;
        ds.PCAExplained{i} = result.PCAExplained;
        ds.PCAExplainedSums(i,:) = fliplr([cumsum([result.PCAExplained(1) result.PCAExplained(2) result.PCAExplained(3)]) 100]);
        ds.PCAScores{i} = result.PCAScores;
    end

    ds.CATFINVaRs(:,4) = sanitize_data(ds.CATFINVaRs(:,4),ds.DatesNum,[],[]);

    [coefficients,scores,explained] = pca_shorthand(ds.CATFINVaRs,false);
    ds.CATFINFirstCoefficients = coefficients(:,1).';
    ds.CATFINFirstExplained = explained(1);

    ds.Indicators(:,1) = sanitize_data(ds.Indicators(:,1),ds.DatesNum,[],[0 1]);
    ds.Indicators(:,2) = scores(:,1);

    r = ds.Returns;
    nan_indices = isnan(r);
    r_m = repmat(mean(r,1,'omitnan'),size(r,1),1);
    r(nan_indices) = r_m(nan_indices);

    [coefficients,scores,explained] = pca_shorthand(r,true);
    ds.PCACoefficientsOverall = coefficients;
    ds.PCAExplainedOverall = explained;
    ds.PCAExplainedSumsOverall = fliplr([cumsum([explained(1) explained(2) explained(3)]) 100]);
    ds.PCAScoresOverall = scores;

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

    labels = regexprep(ds.LabelsCATFINVaRs,'\s\([^)]+\)$','');
    tab = [dates_str array2table(ds.CATFINVaRs,'VariableNames',labels)];
    writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{1},'WriteRowNames',true);

    labels = ds.LabelsIndicatorsSimple;
    tab = [dates_str array2table(ds.Indicators,'VariableNames',labels)];
    writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{2},'WriteRowNames',true);

    labels = ds.LabelsPCAExplained;
    tab = array2table([(1:ds.N).' ds.PCAExplainedOverall],'VariableNames',labels);
    writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{3},'WriteRowNames',true);

    labels = [{'Firms'} ds.FirmNames];
    tab = cell2table([ds.FirmNames.' num2cell(ds.PCACoefficientsOverall)],'VariableNames',labels);
    writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{4},'WriteRowNames',true);

    labels = ds.FirmNames;
    tab = [dates_str array2table(ds.PCAScoresOverall,'VariableNames',labels)];
    writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{5},'WriteRowNames',true);

    worksheets_batch(out,ds.LabelsSheetsSimple,ds.LabelsSheets);

end

%% PLOTTING

function analyze_result(ds)

    safe_plot(@(id)plot_catfin(ds,id));
    safe_plot(@(id)plot_indicators_other(ds,id));
    safe_plot(@(id)plot_pca(ds,id));

end

function plot_catfin(ds,id)

    r = max(0,-ds.CATFINReturns);
    y_limits = plot_limits([-ds.CATFINVaRs r],0.1,0);

    cf = smooth_data(ds.Indicators(:,2));
    cf_label = ['CATFIN (K=' num2str(ds.K * 100) '%, PCA.EV=' sprintf('%.2f%%',ds.CATFINFirstExplained) ')'];

    f = figure('Name','Component Measures > CATFIN','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    subs = gobjects(5,1);

    sub_1 = subplot(2,4,[1 4]);
    plot(sub_1,ds.DatesNum,cf,'Color',[0.000 0.447 0.741]);
    set(sub_1,'XGrid','on','YGrid','on');
    t1 = title(sub_1,cf_label);
    set(t1,'Units','normalized');
    t1_position = get(t1,'Position');
    set(t1,'Position',[0.4783 t1_position(2) t1_position(3)]);

    subs(1) = sub_1;

    for i = 1:4
        sub = subplot(2,4,i + 4);
        plot(sub,ds.DatesNum,r,'Color',[0.000 0.447 0.741]);
        hold on;
            plot(sub,ds.DatesNum,smooth_data(ds.CATFINVaRs(:,i),5),'Color',[1 0.4 0.4],'LineWidth',1.5);
        hold off;
        set(sub,'YLim',y_limits);

        label = ds.LabelsCATFINVaRs{i};

        if (regexp(label,'^[^(]+\([^)]+\)$'))
            title(sub,strrep(strrep(ds.LabelsCATFINVaRs{i},'(','Var ('),')',[ ', PCA.C=' sprintf('%.4f',ds.CATFINFirstCoefficients(i)) ')']));
        else
            title(sub,[ds.LabelsCATFINVaRs{i} ' VaR (PCA.C=' sprintf('%.4f',ds.CATFINFirstCoefficients(i)) ')']);
        end

        subs(i+1) = sub;
    end

    set(subs,'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);

    if (ds.MonthlyTicks)
        date_ticks(subs,'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        date_ticks(subs,'x','yyyy','KeepLimits');
    end

    figure_title('CATFIN');

    pause(0.01);
    frame = get(f,'JavaFrame'); %#ok<JAVFM> 
    set(frame,'Maximized',true);

end

function plot_indicators_other(ds,id)

    alpha = 2 / (ds.BW + 1);

    ar = smooth_data(ds.Indicators(:,1));
    ar_limit = fix(min(ar) * 10) / 10;

    ti = ds.Indicators(:,4);
    ti_ma = [ti(1); filter(alpha,[1 (alpha - 1)],ti(2:end),(1 - alpha) * ti(1))];

    ti_th = NaN(ds.T,1);

    for i = 1:ds.T
        ti_th(i) = quantile(ti(max(1,i-ds.BW):min(ds.T,i+ds.BW)),ds.Q);
    end

    ti_math = ti_ma;
    ti_math(ti_ma <= ti_th) = NaN;

    cs = ds.Indicators(:,3);
    cs_ma = [cs(1); filter(alpha,[1 (alpha - 1)],cs(2:end),(1 - alpha) * cs(1))];

    cs_th = NaN(ds.T,1);

    for i = 1:ds.T
        cs_th(i) = quantile(cs(max(1,i-ds.BW):min(ds.T,i+ds.BW)),ds.Q);
    end

    cs_math = cs_ma;
    cs_math(cs_ma <= cs_th) = NaN;

    f = figure('Name','Component Measures > Other Indicators','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    sub_1 = subplot(2,2,[1 3]);
    plot(sub_1,ds.DatesNum,ar,'Color',[0.000 0.447 0.741]);
    set(sub_1,'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
    set(sub_1,'YLim',[ar_limit 1],'YTick',ar_limit:0.1:1,'YTickLabels',arrayfun(@(x)sprintf('%.f%%',x),(ar_limit:0.1:1) .* 100,'UniformOutput',false));
    set(sub_1,'XGrid','on','YGrid','on');
    t1 = title(sub_1,['Absorption Ratio (F=' num2str(ds.F * 100) '%)']);
    set(t1,'Units','normalized');
    t1_position = get(t1,'Position');
    set(t1,'Position',[0.4783 t1_position(2) t1_position(3)]);

    sub_2 = subplot(2,2,2);
    p2 = plot(sub_2,ds.DatesNum,ti,'Color',[0.65 0.65 0.65]);
    p2.Color(4) = 0.35;
    hold on;
        p21 = plot(sub_2,ds.DatesNum,ti_ma,'Color',[0.000 0.447 0.741],'LineWidth',1);
        p22 = plot(sub_2,ds.DatesNum,ti_math,'Color',[1 0.4 0.4],'LineWidth',1);
    hold off;
    set(sub_2,'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
    l = legend(sub_2,[p21 p22],'EWMA','Threshold Exceeded','Location','best','Orientation','horizontal');
    set(l,'Units','normalized');
    l_position = get(l,'Position');
    set(l,'Position',[0.6710 0.4895 l_position(3) l_position(4)]);
    t2 = title(sub_2,['Turbulence Index (Q=' num2str(ds.Q) ')']);
    set(t2,'Units','normalized');
    t2_position = get(t2,'Position');
    set(t2,'Position',[0.4783 t2_position(2) t2_position(3)]);

    sub_3 = subplot(2,2,4);
    p3 = plot(sub_3,ds.DatesNum,cs,'Color',[0.65 0.65 0.65]);
    p3.Color(4) = 0.35;
    hold on;
        plot(sub_3,ds.DatesNum,cs_ma,'Color',[0.000 0.447 0.741],'LineWidth',1);
        plot(sub_3,ds.DatesNum,cs_math,'Color',[1 0.4 0.4],'LineWidth',1);
    hold off;
    set(sub_3,'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
    t3 = title(sub_3,['Correlation Surprise (Q=' num2str(ds.Q) ')']);
    set(t3,'Units','normalized');
    t3_position = get(t3,'Position');
    set(t3,'Position',[0.4783 t3_position(2) t3_position(3)]);

    if (ds.MonthlyTicks)
        date_ticks([sub_1 sub_2 sub_3],'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        date_ticks([sub_1 sub_2 sub_3],'x','yyyy','KeepLimits');
    end

    figure_title('Other Indicators');

    pause(0.01);
    frame = get(f,'JavaFrame'); %#ok<JAVFM> 
    set(frame,'Maximized',true);

end

function plot_pca(ds,id)

    coefficients = ds.PCACoefficientsOverall(:,1:3);
    [coefficients_rows,coefficients_columns] = size(coefficients);
    [~,indices] = max(abs(coefficients),[],1);
    coefficients_max_len = sqrt(max(sum(coefficients.^2,2)));
    coefficients_columns_sign = sign(coefficients(indices + (0:coefficients_rows:((coefficients_columns-1)*coefficients_rows))));
    coefficients = bsxfun(@times,coefficients,coefficients_columns_sign);

    scores = ds.PCAScoresOverall(:,1:3);
    scores_rows = size(scores,1);
    scores = bsxfun(@times,(coefficients_max_len .* (scores ./ max(abs(scores(:))))),coefficients_columns_sign);

    area_begin = zeros(coefficients_rows,1);
    area_end = NaN(coefficients_rows,1);
    x_area = [area_begin coefficients(:,1) area_end].';
    y_area = [area_begin coefficients(:,2) area_end].';
    z_area = [area_begin coefficients(:,3) area_end].';

    area_end = NaN(scores_rows,1);
    x_points = [scores(:,1) area_end]';
    y_points = [scores(:,2) area_end]';
    z_points = [scores(:,3) area_end]';

    limits_high = 1.1 * max(abs(coefficients(:)));
    limits_low = -limits_high;

    y_ticks = 0:10:100;
    y_tick_labels = arrayfun(@(x)sprintf('%d%%',x),y_ticks,'UniformOutput',false);

    f = figure('Name','Component Measures > Principal Component Analysis','Units','normalized','Tag',id);

    sub_1 = subplot(1,2,1);
    line(x_area(1:2,:),y_area(1:2,:),z_area(1:2,:),'Color',[0 0 1],'LineStyle','-','Marker','none');
    hold on;
        line(x_area(2:3,:),y_area(2:3,:),z_area(2:3,:),'Color',[0 0 1],'LineStyle','none','Marker','.');
        line(x_points,y_points,z_points,'Color',[1 0 0],'LineStyle','none','Marker','.');
        line([limits_low limits_high NaN 0 0 NaN 0 0],[0 0 NaN limits_low limits_high NaN 0 0],[0 0 NaN 0 0 NaN limits_low limits_high],'Color',[0 0 0]);
    hold off;
    view(sub_1,coefficients_columns);
    axis('tight');
    set(sub_1,'XGrid','on','YGrid','on','ZGrid','on');
    xlabel(sub_1,'PC 1');
    ylabel(sub_1,'PC 2');
    zlabel(sub_1,'PC 3');
    title('Coefficients & Scores');

    sub_2 = subplot(1,2,2);
    a1 = area(sub_2,ds.DatesNum,ds.PCAExplainedSums(:,1),'FaceColor',[0.7 0.7 0.7]);
    hold on;
        a2 = area(sub_2,ds.DatesNum,ds.PCAExplainedSums(:,2),'FaceColor','g');
        a3 = area(sub_2,ds.DatesNum,ds.PCAExplainedSums(:,3),'FaceColor','b');
        a4 = area(sub_2,ds.DatesNum,ds.PCAExplainedSums(:,4),'FaceColor','r');
    hold off;
    set([a1 a2 a3 a4],'EdgeColor','none');
    set(sub_2,'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTick',[]);
    set(sub_2,'YLim',[y_ticks(1) y_ticks(end)],'YTick',y_ticks,'YTickLabel',y_tick_labels);
    legend(sub_2,sprintf('PC 4-%d',ds.N),'PC 3','PC 2','PC 1','Location','southeast');
    title('Explained Variance');

    if (ds.MonthlyTicks)
        date_ticks(sub_2,'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        date_ticks(sub_2,'x','yyyy','KeepLimits');
    end

    figure_title('Principal Component Analysis');

    pause(0.01);
    frame = get(f,'JavaFrame'); %#ok<JAVFM> 
    set(frame,'Maximized',true);

end

%% VALIDATION

function out = validate_output(out)

    [path,name,extension] = fileparts(out);

    if (~strcmpi(extension,'.xlsx'))
        out = fullfile(path,[name extension '.xlsx']);
    end

end

function temp = validate_template(temp)

    sheets = {'CATFIN VaRs' 'Indicators' 'PCA Overall Explained' 'PCA Overall Coefficients' 'PCA Overall Scores'};
    file_sheets = validate_xls(temp,'T');

    if (~all(ismember(sheets,file_sheets)))
        error(['The template must contain the following sheets: ' sheets{1} sprintf(', %s',sheets{2:end}) '.']);
    end

    worksheets_batch(temp,sheets);

end
