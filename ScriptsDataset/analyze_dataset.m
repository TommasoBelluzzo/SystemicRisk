% [INPUT]
% ds = A structure representing the dataset.

function analyze_dataset(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('ds',@(x)validateattributes(x,{'struct'},{'nonempty'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    ds = validate_dataset(ipr.ds);

    nargoutchk(0,0);

    analyze_dataset_internal(ds);

end

function analyze_dataset_internal(ds)

    safe_plot(@(id)plot_index(ds,id));

    safe_plot(@(id)plot_boxes(ds,'Returns',id));
    safe_plot(@(id)plot_sequence_returns(ds,id));

    if (~isempty(ds.Volumes))
        safe_plot(@(id)plot_boxes(ds,'Volumes',id));
        safe_plot(@(id)plot_sequence_other(ds,'Volumes',id));
    end

    if (~isempty(ds.Capitalizations))
        safe_plot(@(id)plot_boxes(ds,'Capitalizations',id));
        safe_plot(@(id)plot_sequence_other(ds,'Capitalizations',id));
    end

    if (~isempty(ds.CDS))
        safe_plot(@(id)plot_risk_free_rate(ds,id));
        safe_plot(@(id)plot_boxes(ds,'CDS',id));
        safe_plot(@(id)plot_sequence_other(ds,'CDS',id));
    end

    if (~isempty(ds.Assets) && ~isempty(ds.Equity))
        safe_plot(@(id)plot_boxes(ds,'Assets',id));
        safe_plot(@(id)plot_sequence_other(ds,'Assets',id));

        safe_plot(@(id)plot_boxes(ds,'Equity',id));
        safe_plot(@(id)plot_sequence_other(ds,'Equity',id));

        safe_plot(@(id)plot_boxes(ds,'Liabilities',id));
        safe_plot(@(id)plot_sequence_other(ds,'Liabilities',id));
    end

    if (ds.Crises > 0)
        safe_plot(@(id)plot_crises(ds,id));
    end

end

function plot_boxes(ds,target,id)

    n = ds.N;
    y = ds.(target);

    f = figure('Name',['Dataset > ' target ' (Box Plots)'],'Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);    

    boxplot(y,'Notch','on','Symbol','k.');
    set(findobj(f,'type','line','Tag','Median'),'Color','g');
    set(findobj(f,'-regexp','Tag','\w*Whisker'),'LineStyle','-');
    delete(findobj(f,'-regexp','Tag','\w*Outlier'));

    lower_av = findobj(f,'-regexp','Tag','Lower Adjacent Value');
    lower_av = cell2mat(get(lower_av,'YData'));
    y_low = min(lower_av(:));
    y_low = y_low - abs(y_low / 10);

    upper_av = findobj(f,'-regexp','Tag','Upper Adjacent Value');
    upper_av = cell2mat(get(upper_av,'YData'));
    y_high = max(upper_av(:));
    y_high = y_high + abs(y_high / 10);

    ax = gca();
    set(ax,'TickLength',[0 0]);
    set(ax,'XTick',1:n,'XTickLabels',ds.FirmNames,'XTickLabelRotation',45);
    set(ax,'YLim',[y_low y_high]);

    figure_title(f,target);

    maximize_figure(f);

end

function plot_crises(ds,id)

    f = figure('Name','Dataset > Crises','Units','normalized','Tag',id);

    if (strcmp(ds.CrisesType,'E'))
        cd = ds.CrisesDummy;
        cddn = ds.DatesNum(logical(cd));
        cddn_len = numel(cddn);

        plot(ds.DatesNum,nan(ds.T,1));

        hold on;
            for i = cddn_len:-1:1
                l = line(ones(2,1) .* cddn(i),[0 1],'Color',[1.000 0.400 0.400]);
                set(l,'Tag',num2str(i));
            end
        hold off;

        ax = gca();

        tooltips = ds.CrisisNames;
        tooltips_target = l;
    else
        cd = ds.CrisisDummies;
        k = size(cd,2);

        co = get(gca(),'ColorOrder');
        cor = ceil(k / size(co,1));
        co = repmat(co,cor,1);

        p = gobjects(k,1);

        hold on;
            for i = k:-1:1
                cddn = ds.DatesNum(logical(cd(:,i)));
                cddn_max = max(cddn);
                cddn_min = min(cddn);

                p(i) = patch('XData',[cddn_min cddn_max cddn_max cddn_min],'YData',[0 0 1 1],'EdgeAlpha',0.25,'FaceAlpha',0.40,'FaceColor',co(i,:));
                set(p(i),'Tag',num2str(i));
            end
        hold off;

        ax = gca();

        if (k <= 5)
            legend(ax,p,ds.CrisisNames,'Location','southoutside','Orientation','horizontal');

            tooltips = [];
            tooltips_target = [];
        else
            tooltips = ds.CrisisNames;
            tooltips_target = p;
        end
    end

    set(ax,'Box','on');
    set(ax,'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
    set(ax,'YLim',[0 1],'YTick',[]);

    if (ds.MonthlyTicks)
        date_ticks(ax,'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        date_ticks(ax,'x','yyyy','KeepLimits');
    end

    if (strcmp(ds.CrisesType,'E'))
        figure_title(f,'Crises (Events)');
    else
        figure_title(f,'Crises (Ranges)');
    end

    maximize_figure(f);

    if (~isempty(tooltips))
        dcm = datacursormode(f);
        set(dcm,'Enable','on','SnapToDataVertex','off','UpdateFcn',@(targ,evtd)create_tooltip(targ,evtd,tooltips));
        createDatatip(dcm,tooltips_target,[1 1]);
    end

    function tooltip = create_tooltip(~,evtd,tooltips)

        target = get(evtd,'Target');
        index = str2double(get(target,'Tag'));
        tooltip = tooltips{index};

    end

end

function plot_index(ds,id)

    y = ds.Index;

    yv = y(~isnan(y));
    y_obs = numel(yv);
    y_max = max(yv);
    y_min = min(yv);
    y_avg = mean(yv);
    y_med = median(yv);
    y_std = std(yv);
    y_ske = skewness(yv,0);
    y_kur = kurtosis(yv,0);

    r = yv - y_avg;
    [s_h,s_pval] = shapiro_test(r,0.05);
    [lbq_h,lbq_pval] = lbqtest(r.^2,'Alpha',0.05);
    [a_h,a_pval] = archtest(r,'Alpha',0.05);

    f = figure('Name','Dataset > Index','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    sub_1 = subplot(2,1,1);
    plot(sub_1,ds.DatesNum,ds.Index);
    set(sub_1,'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
    set(sub_1,'YLim',[(y_min - 0.01) (y_max + 0.01)]);
    set(sub_1,'XGrid','on','YGrid','on');
    t1 = title(sub_1,'Log Returns');
    set(t1,'Units','normalized');
    t1_position = get(t1,'Position');
    set(t1,'Position',[0.4783 t1_position(2) t1_position(3)]);

    if (ds.MonthlyTicks)
        date_ticks(sub_1,'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        date_ticks(sub_1,'x','yyyy','KeepLimits');
    end

    sub_2 = subplot(2,1,2);
    hist = histogram(sub_2,ds.Index,50,'FaceColor',[0.749 0.862 0.933],'Normalization','pdf');
    edges = get(hist,'BinEdges');
    edges_max = max(edges);
    edges_min = min(edges);
    [hv,hp] = ksdensity(ds.Index);

    hold on;
        plot(sub_2,hp,hv,'-b','LineWidth',1.5);
    hold off;
    set(sub_2,'XLim',[(edges_min - (edges_min * 0.1)) (edges_max - (edges_max * 0.1))]);
    t2 = title(sub_2,'P&L Distribution');
    set(t2,'Units','normalized');
    t2_position = get(t2,'Position');
    set(t2,'Position',[0.4783 t2_position(2) t2_position(3)]);

    t = figure_title(f,['Index (' ds.IndexName ')']);
    t_position = get(t,'Position');
    set(t,'Position',[t_position(1) -0.0157 t_position(3)]);

    txt_obs = sprintf('Observations: %d',y_obs);
    txt_avg = sprintf('Mean: %.4f',y_avg);
    txt_med = sprintf('Median: %.4f',y_med);
    txt_std = sprintf('Standard Deviation: %.4f',y_std);
    txt_ske = sprintf('Skewness: %.4f',y_ske);
    txt_kur = sprintf('Kurtosis: %.4f',y_kur);

    if (s_h)
        txt_s = sprintf('Shapiro Test: T (%.4f)',s_pval);
    else
        txt_s = sprintf('Shapiro Test: F (%.4f)',s_pval);
    end

    if (lbq_h)
        txt_lbq = sprintf('LBQ Test: T (%.4f)',lbq_pval);
    else
        txt_lbq = sprintf('LBQ Test: F (%.4f)',lbq_pval);
    end

    if (a_h)
        txt_a = sprintf('ARCH Test: T (%.4f)',a_pval);
    else
        txt_a = sprintf('ARCH Test: F (%.4f)',a_pval);
    end

    txt = {txt_obs '' txt_avg txt_med txt_std txt_ske txt_kur '' txt_s txt_lbq txt_a};

    annotation('TextBox',(get(sub_2,'Position') + [0.01 -0.025 0 0]),'String',txt,'EdgeColor','none','FitBoxToText','on','FontSize',8);

    maximize_figure(f);

end

function plot_risk_free_rate(ds,id)

    rfr = ds.RiskFreeRate;
    y_limits_rfr = plot_limits(rfr,0.1);

    rfr_pc = [0; (((rfr(2:end) - rfr(1:end-1)) ./ rfr(1:end-1)) .* 100)];
    y_limits_rfr_pc = plot_limits(rfr_pc,0.1);

    f = figure('Name','Dataset > Risk-Free Rate','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    sub_1 = subplot(2,1,1);
    plot(sub_1,ds.DatesNum,smooth_data(rfr));
    set(sub_1,'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
    set(sub_1,'YLim',y_limits_rfr);
    set(sub_1,'XGrid','on','YGrid','on');
    t1 = title(sub_1,'Trend');
    set(t1,'Units','normalized');
    t1_position = get(t1,'Position');
    set(t1,'Position',[0.4783 t1_position(2) t1_position(3)]);

    sub_2 = subplot(2,1,2);
    plot(sub_2,ds.DatesNum,rfr_pc);
    set(sub_2,'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
    set(sub_2,'YLim',y_limits_rfr_pc);
    set(sub_2,'YTickLabels',arrayfun(@(x)sprintf('%.f%%',x),get(sub_2,'YTick'),'UniformOutput',false));
    set(sub_2,'XGrid','on','YGrid','on');
    t2 = title(sub_2,'Percent Change');
    set(t2,'Units','normalized');
    t2_position = get(t2,'Position');
    set(t2,'Position',[0.4783 t2_position(2) t2_position(3)]);

    if (ds.MonthlyTicks)
        date_ticks([sub_1 sub_2],'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        date_ticks([sub_1 sub_2],'x','yyyy','KeepLimits');
    end

    figure_title(f,'Risk-Free Rate');

    maximize_figure(f);

end

function plot_sequence_returns(ds,id)

    n = ds.N;
    t = ds.T;
    dn = ds.DatesNum;
    mt = ds.MonthlyTicks;

    ts = ds.Returns;
    data = [repmat({dn},1,n); mat2cell(ts,t,ones(1,n))];

    plots_title = [repmat({'Log Returns'},1,n); repmat({'P&L Distribution'},1,n)];

    x_limits = [dn(1) dn(end)];

    core = struct();

    core.N = n;
    core.Data = data;
    core.Function = @(subs,data)plot_function(subs,data);

    core.OuterTitle = 'Dataset > Returns (Time Series)';
    core.InnerTitle = 'Returns (Time Series)';
    core.SequenceTitles = ds.FirmNames;

    core.PlotsAllocation = [2 1];
    core.PlotsSpan = {1 2};
    core.PlotsTitle = plots_title;

    core.XDates = {mt []};
    core.XGrid = {true false};
    core.XLabel = {[] []};
    core.XLimits = {x_limits []};
    core.XRotation = {45 []};
    core.XTick = {[] []};
    core.XTickLabels = {[] []};

    core.YGrid = {true false};
    core.YLabel = {[] []};
    core.YLimits = {[] []};
    core.YRotation = {[] []};
    core.YTick = {[] []};
    core.YTickLabels = {[] []};

    sequential_plot(core,id);

    function plot_function(subs,data)

        x = data{1};
        y = data{2};

        yv = y(~isnan(y));
        y_obs = numel(yv);
        y_max = max(yv);
        y_min = min(yv);
        y_avg = mean(yv);
        y_med = median(yv);
        y_std = std(yv);
        y_ske = skewness(yv,0);
        y_kur = kurtosis(yv,0);

        r = yv - y_avg;
        [s_h,s_pval] = shapiro_test(r,0.05);
        [lbq_h,lbq_pval] = lbqtest(r.^2,'Alpha',0.05);
        [a_h,a_pval] = archtest(r,'Alpha',0.05);

        d = find(isnan(y),1,'first');

        if (isempty(d))
            xd = [];
        else
            xd = x(d) - 1;
        end

        delete(findall(gcf,'type','annotation'));

		sub_1 = subs(1);
        plot(sub_1,x,y,'Color',[0.000 0.447 0.741]);
        set(sub_1,'YLim',[(y_min - 0.01) (y_max + 0.01)]);

        if (~isempty(xd))
            hold(sub_1,'on');
                plot(sub_1,[xd xd],get(sub_1,'YLim'),'Color',[1.000 0.400 0.400]);
            hold(sub_1,'off');
        end

		sub_2 = subs(2);
        hist = histogram(sub_2,y,50,'FaceColor',[0.749 0.862 0.933],'Normalization','pdf');
        edges = get(hist,'BinEdges');
        edges_max = max(edges);
        edges_min = min(edges);
        [dv,dp] = ksdensity(y);

        hold(sub_2,'on');
            plot(sub_2,dp,dv,'-b','LineWidth',1.5);
        hold(sub_2,'off');
        set(sub_2,'XLim',[(edges_min - (edges_min * 0.1)) (edges_max - (edges_max * 0.1))]);

        txt_obs = sprintf('Observations: %d',y_obs);
        txt_avg = sprintf('Mean: %.4f',y_avg);
        txt_med = sprintf('Median: %.4f',y_med);
        txt_std = sprintf('Standard Deviation: %.4f',y_std);
        txt_ske = sprintf('Skewness: %.4f',y_ske);
        txt_kur = sprintf('Kurtosis: %.4f',y_kur);

        if (s_h)
            txt_s = sprintf('Shapiro Test: T (%.4f)',s_pval);
        else
            txt_s = sprintf('Shapiro Test: F (%.4f)',s_pval);
        end

        if (lbq_h)
            txt_lbq = sprintf('LBQ Test: T (%.4f)',lbq_pval);
        else
            txt_lbq = sprintf('LBQ Test: F (%.4f)',lbq_pval);
        end

        if (a_h)
            txt_a = sprintf('ARCH Test: T (%.4f)',a_pval);
        else
            txt_a = sprintf('ARCH Test: F (%.4f)',a_pval);
        end

        txt = {txt_obs '' txt_avg txt_med txt_std txt_ske txt_kur '' txt_s txt_lbq txt_a};

        annotation('TextBox',(get(sub_2,'Position') + [0.01 -0.025 0 0]),'String',txt,'EdgeColor','none','FitBoxToText','on','FontSize',8);

    end

end

function plot_sequence_other(ds,target,id)

    n = ds.N;
    t = ds.T;
    dn = ds.DatesNum;
    mt = ds.MonthlyTicks;

    ts = smooth_data(ds.(target));
    data = [repmat({dn},1,n); mat2cell(ts,t,ones(1,n))];

    plots_title = repmat({' '},1,n);

    x_limits = [dn(1) dn(end)];

    core = struct();

    core.N = n;
    core.Data = data;
    core.Function = @(subs,data)plot_function(subs,data);

    core.OuterTitle = ['Dataset > ' target ' (Time Series)'];
    core.InnerTitle = [target ' (Time Series)'];
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
    core.YLimits = {[]};
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

		sub_1 = subs(1);
        plot(sub_1,x,y,'Color',[0.000 0.447 0.741]);

        if (~isempty(xd))
            hold(sub_1,'on');
                plot(sub_1,[xd xd],get(sub_1,'YLim'),'Color',[1.000 0.400 0.400]);
            hold(sub_1,'off');
        end

    end

end
