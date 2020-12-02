% [INPUT]
% ds = A structure representing the dataset.
% sn = A string representing the serial number of the result file.
% temp = A string representing the full path to the Excel spreadsheet used as template for the result file.
% out = A string representing the full path to the Excel spreadsheet to which the results are written, eventually replacing the previous ones.
% k = A float [0.90,0.99] representing the confidence level (optional, default=0.95).
% d = A float [0.1,0.6] representing the crisis threshold for the market index decline used to calculate the LRMES (optional, default=0.4).
% car = A float [0.03,0.20] representing the capital adequacy ratio used to calculate SES and SRISK (optional, default=0.08).
% sf = A float [0,1] representing the fraction of separate accounts, if available, to include in liabilities and used to calculate SES and SRISK (optional, default=0.40).
% fr = An integer [0,6] representing the number of months of forward-rolling used to calculate the SRISK, simulating the difficulty of renegotiating debt in case of financial distress (optional, default=3).
% analyze = A boolean that indicates whether to analyse the results and display plots (optional, default=false).
%
% [OUTPUT]
% result = A structure representing the original dataset inclusive of intermediate and final calculations.
% stopped = A boolean that indicates whether the process has been stopped through user input.

function [result,stopped] = run_cross_sectional(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('ds',@(x)validateattributes(x,{'struct'},{'nonempty'}));
        ip.addRequired('sn',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addRequired('temp',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addRequired('out',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addOptional('k',0.95,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.90 '<=' 0.99 'scalar'}));
        ip.addOptional('d',0.40,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.1 '<=' 0.6 'scalar'}));
        ip.addOptional('car',0.08,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.03 '<=' 0.20 'scalar'}));
        ip.addOptional('sf',0.40,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0 '<=' 1 'scalar'}));
        ip.addOptional('fr',3,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 0 '<=' 6 'scalar'}));
        ip.addOptional('analyze',false,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    ds = validate_dataset(ipr.ds,'CrossSectional');
    sn = ipr.sn;
    temp = validate_template(ipr.temp);
    out = validate_output(ipr.out);
    k = ipr.k;
    d = ipr.d;
    car = ipr.car;
    sf = ipr.sf;
    fr = ipr.fr;
    analyze = ipr.analyze;

    nargoutchk(1,2);

    [result,stopped] = run_cross_sectional_internal(ds,sn,temp,out,k,d,car,sf,fr,analyze);

end

function [result,stopped] = run_cross_sectional_internal(ds,sn,temp,out,k,d,car,sf,fr,analyze)

    result = [];
    stopped = false;
    e = [];

    ds = initialize(ds,sn,k,d,car,sf,fr);
    n = ds.N;
    t = ds.T;

    bar = waitbar(0,'Initializing cross-sectional measures...','CreateCancelBtn',@(src,event)setappdata(gcbf(),'Stop', true));
    setappdata(bar,'Stop',false);
    cleanup = onCleanup(@()delete(bar));

    pause(1);
    waitbar(0,bar,'Calculating cross-sectional measures...');
    pause(1);

    try

        rm = ds.Index;
        rf = ds.Returns;

        cp = ds.Capitalizations;
        lb = ds.TargetLiabilities;
        lbr = ds.TargetLiabilitiesRolled;
        sv = ds.StateVariables;

        for i = 1:n
            waitbar((i - 1) / n,bar,['Calculating cross-sectional measures for ' ds.FirmNames{i} '...']);

            if (getappdata(bar,'Stop'))
                stopped = true;
                break;
            end

            offset = min(ds.Defaults(i) - 1,t);

            r_i = [rm(1:offset) rf(1:offset,i)];
            cp_i = cp(1:offset,i);
            lb_i = lb(1:offset,i);
            lbr_i = lbr(1:offset,i);

            if (isempty(sv))
                sv_i = [];
            else
                sv_i = sv(1:offset,:);
            end

            [beta,var,es,covar,dcovar,mes,ses,srisk] = cross_sectional_metrics(r_i,cp_i,lb_i,lbr_i,sv_i,ds.A,ds.D,ds.CAR);
            ds.Beta(1:offset,i) = beta;
            ds.VaR(1:offset,i) = var;
            ds.ES(1:offset,i) = es;
            ds.CoVaR(1:offset,i) = covar;
            ds.DeltaCoVaR(1:offset,i) = dcovar;
            ds.MES(1:offset,i) = mes;
            ds.SES(1:offset,i) = ses;
            ds.SRISK(1:offset,i) = srisk;

            [caviar,~,ir_fm,ir_mf] = bivariate_caviar(r_i,ds.A);
            ds.CAViaR(1:offset,i) = caviar;
            ds.CAViaRIRFM{i} = ir_fm;
            ds.CAViaRIRMF{i} = ir_mf;

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
    waitbar(1,bar,'Writing cross-sectional measures...');
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

function ds = initialize(ds,sn,k,d,car,sf,fr)

    lb = ds.Liabilities;
    sa = ds.SeparateAccounts;

    lbr = forward_roll_data(lb,ds.DatesNum,fr);

    if (~isempty(sa))
        lb = lb - ((1 - sf) .* sa);

        sar = forward_roll_data(sa,ds.DatesNum,fr);
        lbr = lbr - ((1 - sf) .* sar);
    end

    n = ds.N;
    t = ds.T;

    ds.Result = 'CrossSectional';
    ds.ResultDate = now();
    ds.ResultAnalysis = @(ds)analyze_result(ds);
    ds.ResultSerial = sn;

    ds.A = 1 - k;
    ds.CAR = car;
    ds.D = d;
    ds.FR = fr;
    ds.K = k;
    ds.SF = sf;

    k_all_label = [' (K=' num2str(ds.K * 100) '%)'];
    ses_label =  [' (CAR=' num2str(ds.CAR * 100) '%)'];
    srisk_label = [' (D=' num2str(ds.D * 100) '%, CAR=' num2str(ds.CAR * 100) '%)'];

    ds.LabelsMeasuresSimple = {'Beta' 'VaR' 'ES' 'CAViaR' 'CoVaR' 'Delta CoVaR' 'MES' 'SES' 'SRISK'};
    ds.LabelsMeasures = {'Beta' ['VaR' k_all_label] ['ES' k_all_label] ['CAViaR' k_all_label] ['CoVaR' k_all_label] ['Delta CoVaR' k_all_label] ['MES' k_all_label] ['SES' ses_label] ['SRISK' srisk_label]};

    ds.LabelsSheetsSimple = [ds.LabelsMeasuresSimple {'Averages'}];
    ds.LabelsSheets = [ds.LabelsMeasures {'Averages'}];

    ds.TargetLiabilities = lb;
    ds.TargetLiabilitiesRolled = lbr;

    m = numel(ds.LabelsMeasuresSimple);

    ds.CAViaRIRFM = cell(m,1);
    ds.CAViaRIRMF = cell(m,1);

    ds.Beta = NaN(t,n);
    ds.VaR = NaN(t,n);
    ds.ES = NaN(t,n);
    ds.CAViaR = NaN(t,n);
    ds.CoVaR = NaN(t,n);
    ds.DeltaCoVaR = NaN(t,n);
    ds.MES = NaN(t,n);
    ds.SES = NaN(t,n);
    ds.SRISK = NaN(t,n);
    ds.Averages = NaN(t,m);

    ds.RankingConcordance = NaN(m);
    ds.RankingStability = NaN(1,m);

    ds.ComparisonReferences = {'Averages' 4:9 strcat({'CS-'},strrep(ds.LabelsMeasuresSimple(4:end),'Delta ','D'))};

end

function ds = finalize(ds)

    n = ds.N;

    weights = max(0,ds.Capitalizations ./ repmat(sum(ds.Capitalizations,2,'omitnan'),1,n));

    beta_avg = sum(ds.Beta .* weights,2,'omitnan');
    var_avg = sum(ds.VaR .* weights,2,'omitnan');
    es_avg = sum(ds.ES .* weights,2,'omitnan');
    caviar_avg = sum(ds.CAViaR .* weights,2,'omitnan');
    covar_avg = sum(ds.CoVaR .* weights,2,'omitnan');
    dcovar_avg = sum(ds.DeltaCoVaR .* weights,2,'omitnan');
    mes_avg = sum(ds.MES .* weights,2,'omitnan');
    ses_avg = sum(ds.SES .* weights,2,'omitnan');
    srisk_avg = sum(ds.SRISK .* weights,2,'omitnan');
    ds.Averages = [beta_avg var_avg es_avg caviar_avg covar_avg dcovar_avg mes_avg ses_avg srisk_avg];

    measures_len = numel(ds.LabelsMeasuresSimple);
    measures = cell(measures_len,1);

    for i = 1:measures_len
        measures{i} = ds.(strrep(ds.LabelsMeasuresSimple{i},' ',''));
    end

    [rc,rs] = kendall_rankings(measures);
    ds.RankingConcordance = rc;
    ds.RankingStability = rs;

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

    tab = [dates_str array2table(ds.Averages,'VariableNames',strrep(ds.LabelsSheetsSimple(1:end-1),' ','_'))];
    writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{end},'WriteRowNames',true);

    worksheets_batch(out,ds.LabelsSheetsSimple,ds.LabelsSheets);

end

%% PLOTTING

function analyze_result(ds)

    safe_plot(@(id)plot_sequence_other(ds,'Beta',id));
    safe_plot(@(id)plot_sequence_other(ds,'VaR',id));
    safe_plot(@(id)plot_sequence_other(ds,'ES',id));
    safe_plot(@(id)plot_sequence_caviar(ds,id));
    safe_plot(@(id)plot_sequence_other(ds,'CoVaR',id));
    safe_plot(@(id)plot_sequence_other(ds,'Delta CoVaR',id));
    safe_plot(@(id)plot_sequence_other(ds,'MES',id));
    safe_plot(@(id)plot_sequence_other(ds,'SES',id));
    safe_plot(@(id)plot_sequence_other(ds,'SRISK',id));
    safe_plot(@(id)plot_idiosyncratic_averages(ds,id));
    safe_plot(@(id)plot_systemic_averages(ds,id));
    safe_plot(@(id)plot_correlations(ds,id));
    safe_plot(@(id)plot_rankings(ds,id));

end

function [ax,big_ax] = gplotmatrix_stable(f,x,labels)

    n = size(x,2);

    clf(f);
    big_ax = newplot();
    hold_state = ishold();

    set(big_ax,'Color','none','Parent',f,'Visible','off');

    position = get(big_ax,'Position');
    width = position(3) / n;
    height = position(4) / n;
    position(1:2) = position(1:2) + (0.02 .* [width height]);

    [m,~,k] = size(x);

    x_min = min(x,[],1);
    x_max = max(x,[],1);
    x_limits = repmat(cat(3,x_min,x_max),[n 1 1]);
    y_limits = repmat(cat(3,x_min.',x_max.'),[1 n 1]);

    for i = n:-1:1
        for j = 1:1:n
            ax_position = [(position(1) + (j - 1) * width) (position(2) + (n - i) * height) (width * 0.98) (height * 0.98)];
            ax1(i,j) = axes('Box','on','Parent',f,'Position',ax_position,'Visible','on');

            if (i == j)
                ax2(j) = axes('Parent',f,'Position',ax_position);
                histogram(reshape(x(:,i,:),[m k]),'BinMethod','scott','DisplayStyle','bar','FaceColor',[0.678 0.922 1],'Norm','pdf');
                set(ax2(j),'YAxisLocation','right','XGrid','off','XTick',[],'XTickLabel','');
                set(ax2(j),'YGrid','off','YLim',get(ax2(j),'YLim') .* [1 1.05],'YTick',[],'YTickLabel','');
                set(ax2(j),'Visible','off');
                axis(ax2(j),'tight');
                x_limits(i,j,:) = get(ax2(j),'XLim');      
            else
                iscatter(reshape(x(:,j,:),[m k]),reshape(x(:,i,:),[m k]),ones(size(x,1),1),[0 0 1],'o',2);
                axis(ax1(i,j),'tight');
                x_limits(i,j,:) = get(ax1(i,j),'XLim');
                y_limits(i,j,:) = get(ax1(i,j),'YLim');
            end

            set(ax1(i,j),'XGrid','off','XLimMode','auto','YGrid','off','YLimMode','auto');
        end
    end

    x_limits_min = min(x_limits(:,:,1),[],1);
    x_limits_max = max(x_limits(:,:,2),[],1);

    y_limits_min = min(y_limits(:,:,1),[],2);
    y_limits_max = max(y_limits(:,:,2),[],2);

    for i = 1:n
        set(ax1(i,1),'YLim',[y_limits_min(i,1) y_limits_max(i,1)]);
        dy = diff(get(ax1(i,1),'YLim')) * 0.05;
        set(ax1(i,:),'YLim',[(y_limits_min(i,1)-dy) y_limits_max(i,1)+dy]);

        set(ax1(1,i),'XLim',[x_limits_min(1,i) x_limits_max(1,i)])
        dx = diff(get(ax1(1,i),'XLim')) * 0.05;
        set(ax1(:,i),'XLim',[(x_limits_min(1,i) - dx) (x_limits_max(1,i) + dx)])
        set(ax2(i),'XLim',[(x_limits_min(1,i) - dx) (x_limits_max(1,i) + dx)])
    end

    for i = 1:n
        set(get(ax1(i,1),'YLabel'),'String',labels{i});
        set(get(ax1(n,i),'XLabel'),'String',labels{i});
    end

    set(ax1(1:n-1,:),'XTickLabel','');
    set(ax1(:,2:n),'YTickLabel','');

    set(f,'CurrentAx',big_ax);
    set([get(big_ax,'Title'); get(big_ax,'XLabel'); get(big_ax,'YLabel')],'String','','Visible','on');

    if (~hold_state)
        set(f,'NextPlot','replace')
    end

    for i = 1:n
        hz = zoom();

        linkprop(ax1(i,:),{'YLim' 'YScale'});
        linkprop(ax1(:,i),{'XLim' 'XScale'});

        setAxesZoomMotion(hz,ax2(i),'horizontal');        
    end

    set(pan(),'ActionPreCallback',@size_changed_callback);

    ax = [ax1; ax2(:).'];

    function size_changed_callback(~,~)

        if (~all(isgraphics(ax1(:))))
            return;
        end

        set(ax1(1:n,1),'YTickLabelMode','auto');
        set(ax1(n,1:n),'XTickLabelMode','auto');

    end

end

function plot_idiosyncratic_averages(ds,id)

    averages = ds.Averages(:,1:3);
    beta = averages(:,1);
    others = averages(:,2:3);

    y_limits_beta = plot_limits(beta,0.1,0);
    y_limits_others = plot_limits(others,0.1);

    f = figure('Name','Cross-Sectional Measures > Idiosyncratic Averages','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    sub_1 = subplot(2,2,[1 3]);
    plot(sub_1,ds.DatesNum,smooth_data(beta),'Color',[0.000 0.447 0.741]);
    set(sub_1,'YLim',y_limits_beta);
    title(sub_1,ds.LabelsMeasures{1});

    sub_2 = subplot(2,2,2);
    plot(sub_2,ds.DatesNum,smooth_data(averages(:,2)),'Color',[0.000 0.447 0.741]);
    set(sub_2,'YLim',y_limits_others);
    title(sub_2,ds.LabelsMeasures{2});

    sub_3 = subplot(2,2,4);
    plot(sub_3,ds.DatesNum,smooth_data(averages(:,3)),'Color',[0.000 0.447 0.741]);
    set(sub_3,'YLim',y_limits_others,'YTick',get(sub_2,'YTick'),'YTickLabel',get(sub_2,'YTickLabel'),'YTickLabelMode',get(sub_2,'YTickLabelMode'),'YTickMode',get(sub_2,'YTickMode'));
    title(sub_3,ds.LabelsMeasures{3});

    set([sub_1 sub_2 sub_3],'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
    set([sub_1 sub_2 sub_3],'XGrid','on','YGrid','on');

    if (ds.MonthlyTicks)
        date_ticks([sub_1 sub_2 sub_3],'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        date_ticks([sub_1 sub_2 sub_3],'x','yyyy','KeepLimits');
    end

    figure_title('Idiosyncratic Averages');

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_systemic_averages(ds,id)

    y_limits = zeros(6,2);

    averages_quantile = ds.Averages(:,4:7);
    y_limits(1:4,:) = repmat(plot_limits(averages_quantile,0.1),4,1);

    averages_volume = ds.Averages(:,8:9);
    y_limits(5:6,:) = repmat(plot_limits(averages_volume,0.1),2,1);

    subplot_offsets = [1; 3; 5; 2; 4; 6];

    f = figure('Name','Cross-Sectional Measures > Systemic Averages','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    subs = gobjects(6,1);

    for i = 1:6
        sub = subplot(3,2,subplot_offsets(i));
        plot(sub,ds.DatesNum,smooth_data(ds.Averages(:,i+3)),'Color',[0.000 0.447 0.741]);
        set(sub,'YLim',y_limits(i,:));
        title(sub,ds.LabelsMeasures{i+3});

        subs(i) = sub;
    end

    set(subs,'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
    set(subs,'XGrid','on','YGrid','on');

    if (ds.MonthlyTicks)
        date_ticks(subs,'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        date_ticks(subs,'x','yyyy','KeepLimits');
    end

    y_ticks = get(subs(1),'YTick');
    y_tick_labels = arrayfun(@(x)sprintf('%.2f',x),y_ticks,'UniformOutput',false);
    set(subs(1:4),'YTick',y_ticks,'YTickLabel',y_tick_labels);

    y_ticks = get(subs(5),'YTick');
    y_tick_labels = arrayfun(@(x)sprintf('%.0f',x),y_ticks,'UniformOutput',false);
    set(subs(5:6),'YTick',y_ticks,'YTickLabel',y_tick_labels);

    figure_title('Systemic Averages');

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_correlations(ds,id)

    mu = mean(ds.Averages,1);
    sigma = std(ds.Averages,1);

    [rho,pval] = corr(ds.Averages);
    rho(isnan(rho)) = 0;

    z = bsxfun(@minus,ds.Averages,mu);
    z = bsxfun(@rdivide,z,sigma);
    z_limits = [nanmin(z(:)) nanmax(z(:))];

    n = numel(ds.LabelsMeasures);

    f = figure('Name','Cross-Sectional Measures > Correlation Matrix','Units','normalized','Tag',id);

    [ax,big_ax] = gplotmatrix_stable(f,ds.Averages,ds.LabelsMeasuresSimple);

    x_labels = get(ax,'XLabel');
    y_labels = get(ax,'YLabel');
    set([x_labels{:}; y_labels{:}],'FontWeight','bold');

    x_labels_grey = cellfun(@(x)x.String,x_labels,'UniformOutput',false);
    x_labels_grey_indices = ismember(x_labels_grey,ds.LabelsMeasuresSimple(1:3));
    y_labels_grey = cellfun(@(x)x.String,y_labels,'UniformOutput',false);
    y_labels_grey_indices = ismember(y_labels_grey,ds.LabelsMeasuresSimple(1:3));
    set([x_labels{x_labels_grey_indices}; y_labels{y_labels_grey_indices}],'Color',[0.5 0.5 0.5]);

    for i = 1:n
        for j = 1:n
            ax_ij = ax(i,j);

            z_limits_current = 1.1 .* z_limits;
            x_limits = mu(j) + (z_limits_current * sigma(j));
            y_limits = mu(i) + (z_limits_current * sigma(i));

            set(get(big_ax,'Parent'),'CurrentAxes',ax_ij);
            set(ax_ij,'XLim',x_limits,'XTick',[]);
            set(ax_ij,'YLim',y_limits,'YTick',[]);
            axis(ax_ij,'normal');

            if (i ~= j)
                line = lsline();
                set(line,'Color','r');

                if (pval(i,j) < 0.05)
                    color = 'r';
                else
                    color = 'k';
                end

                annotation('TextBox',get(ax_ij,'Position'),'String',num2str(rho(i,j),'%.2f'),'Color',color,'EdgeColor','none','FontWeight','Bold');
            end
        end
    end

    annotation('TextBox',[0 0 1 1],'String','Correlation Matrix','EdgeColor','none','FontName','Helvetica','FontSize',14,'HorizontalAlignment','center');

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

    f = figure('Name','Cross-Sectional Measures > Rankings','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    sub_1 = subplot(1,2,1);
    bar(sub_1,seq,rs,'FaceColor',[0.749 0.862 0.933]);
    set(sub_1,'XTickLabel',rs_names,'XTickLabelRotation',45);
    set(sub_1,'YLim',[0 1]);
    title(sub_1,'Ranking Stability');

    if (~verLessThan('MATLAB','8.4'))
        tl = get(sub_1,'XTickLabel');
        tl_new = cell(size(tl));

        for i = 1:length(tl)
            tl_i = tl{i};

            if (ismember(tl_i,labels(1:3)))
                tl_new{i} = ['\color[rgb]{0.5 0.5 0.5}\bf{' tl_i '}'];
            else
                tl_new{i} = ['\bf{' tl_i '}'];
            end
        end

        set(sub_1,'XTickLabel',tl_new);
    end

    sub_2 = subplot(1,2,2);
    pcolor(padarray(rc,[1 1],'post'));
    colormap([1 1 1; 0.65 0.65 0.65; 0.749 0.862 0.933]);
    axis('image');
    text(rc_x,rc_y,rc_text,'FontSize',9,'HorizontalAlignment','center');
    set(sub_2,'FontWeight','bold','TickLength',[0 0]);
    set(sub_2,'XAxisLocation','bottom','XTick',off,'XTickLabels',labels,'XTickLabelRotation',45);
    set(sub_2,'YDir','reverse','YTick',off,'YTickLabels',labels,'YTickLabelRotation',45);
    title(sub_2,'Ranking Concordance');

    if (~verLessThan('MATLAB','8.4'))
        tl = get(sub_2,'XTickLabel');
        tl_new = cell(size(tl));

        for i = 1:length(tl)
            tl_i = tl{i};

            if (ismember(tl_i,labels(1:3)))
                tl_new{i} = ['\color[rgb]{0.5 0.5 0.5}\bf{' tl_i '}'];
            else
                tl_new{i} = ['\bf{' tl_i '}'];
            end
        end

        set(sub_2,'XTickLabel',tl_new);

        tl = get(sub_2,'YTickLabel');
        tl_new = cell(size(tl));

        for i = 1:length(tl)
            tl_i = tl{i};

            if (ismember(tl_i,labels(1:3)))
                tl_new{i} = ['\color[rgb]{0.5 0.5 0.5}\bf{' tl_i '} '];
            else
                tl_new{i} = ['\bf{' tl_i '} '];
            end
        end

        set(sub_2,'YTickLabel',tl_new);
    end

    figure_title('Rankings (Kendall''s W)');

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_sequence_caviar(ds,id)

    n = ds.N;
    t = ds.T;
    dn = ds.DatesNum;
    mt = ds.MonthlyTicks;

    ts = smooth_data(ds.CAViaR);

    data = [repmat({dn},1,n); mat2cell(ts,t,ones(1,n)); repmat({1:200},1,n); ds.CAViaRIRFM.'; ds.CAViaRIRMF.'];

    [~,index] = ismember('CAViaR',ds.LabelsMeasuresSimple);
    plots_title = cell(3,20);
    plots_title(1,:) = repmat(ds.LabelsMeasures(index),1,n);
    plots_title(2,:) = repmat({'Impulse Response - Firm on Market Shock'},1,n);
    plots_title(3,:) = repmat({'Impulse Response - Market on Firm Shock'},1,n);

    x_limits = {[dn(1) dn(end)] [1 200] [1 200]};
    y_limits = {plot_limits(ts,0.1) [] []};

    core = struct();

    core.N = n;
    core.Data = data;
    core.Function = @(subs,data)plot_function(subs,data);

    core.OuterTitle = 'Cross-Sectional Measures > CAViaR Time Series';
    core.InnerTitle = 'CAViaR Time Series';
    core.SequenceTitles = ds.FirmNames;

    core.PlotsAllocation = [2 2];
    core.PlotsSpan = {[1 2] 3 4};
    core.PlotsTitle = plots_title;

    core.XDates = {mt [] []};
    core.XGrid = {true true true};
    core.XLabel = {[] [] []};
    core.XLimits = x_limits;
    core.XRotation = {45 [] []};
    core.XTick = {[] [] []};
    core.XTickLabels = {[] [] []};

    core.YGrid = {true true true};
    core.YLabel = {[] [] []};
    core.YLimits = y_limits;
    core.YRotation = {[] [] []};
    core.YTick = {[] [] []};
    core.YTickLabels = {[] [] []};

    sequential_plot(core,id);

    function plot_function(subs,data)

        x_caviar = data{1};
        caviar = data{2};

        x_ir = data{3};
        ir_fm = data{4};
        ir_mf = data{5};

        d = find(isnan(caviar),1,'first');

        if (isempty(d))
            xd = [];
        else
            xd = x_caviar(d) - 1;
        end

        plot(subs(1),x_caviar,caviar,'Color',[0.000 0.447 0.741]);

        if (~isempty(xd))
            hold(subs(1),'on');
                plot(subs(1),[xd xd],get(subs(1),'YLim'),'Color',[1 0.4 0.4]);
            hold(subs(1),'off');
        end

        plot(subs(2),x_ir,ir_fm(:,1),'Color',[0.000 0.447 0.741]);
        hold(subs(2),'on');
            plot(subs(2),x_ir,ir_fm(:,2),'Color',[1 0.4 0.4],'LineStyle','--');
            plot(subs(2),x_ir,ir_fm(:,3),'Color',[1 0.4 0.4],'LineStyle','--');
        hold(subs(2),'off');

        plot(subs(3),x_ir,ir_mf(:,1),'Color',[0.000 0.447 0.741]);
        hold(subs(3),'on');
            plot(subs(3),x_ir,ir_mf(:,2),'Color',[1 0.4 0.4],'LineStyle','--');
            plot(subs(3),x_ir,ir_mf(:,3),'Color',[1 0.4 0.4],'LineStyle','--');
        hold(subs(3),'off');

    end

end

function plot_sequence_other(ds,target,id)

    n = ds.N;
    t = ds.T;
    dn = ds.DatesNum;
    mt = ds.MonthlyTicks;

    ts = smooth_data(ds.(strrep(target,' ','')));

    data = [repmat({dn},1,n); mat2cell(ts,t,ones(1,n))];

    [~,index] = ismember(target,ds.LabelsMeasuresSimple);
    plots_title = repmat(ds.LabelsMeasures(index),1,n);

    x_limits = [dn(1) dn(end)];
    y_limits = plot_limits(ts,0.1);

    core = struct();

    core.N = n;
    core.Data = data;
    core.Function = @(subs,data)plot_function(subs,data);

    core.OuterTitle = ['Cross-Sectional Measures > ' target ' Time Series'];
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

%% VALIDATION

function out = validate_output(out)

    [path,name,extension] = fileparts(out);

    if (~strcmpi(extension,'.xlsx'))
        out = fullfile(path,[name extension '.xlsx']);
    end

end

function temp = validate_template(temp)

    sheets = {'Beta' 'VaR' 'ES' 'CAViaR' 'CoVaR' 'Delta CoVaR' 'MES' 'SES' 'SRISK' 'Averages'};
    file_sheets = validate_xls(temp,'T');

    if (~all(ismember(sheets,file_sheets)))
        error(['The template must contain the following sheets: ' sheets{1} sprintf(', %s',sheets{2:end}) '.']);
    end

    worksheets_batch(temp,sheets);

end
