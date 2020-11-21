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

    figure_title(target);

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

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
                l = line(ones(2,1) .* cddn(i),[0 1],'Color',[1 0.4 0.4]);
                set(l,'Tag',num2str(i));
            end
        hold off;

        ax = gca();

        tooltips = ds.CrisisNames;
        tooltips_target = l;
    else
        cd = ds.CrisisDummies;
        k = size(cd,2);

        co = get(gca,'ColorOrder');
        cor = ceil(k / size(co,1));
        co = repmat(co,cor,1);

        hold on;
            for i = k:-1:1
                cddn = ds.DatesNum(logical(cd(:,i)));
                cddn_max = max(cddn);
                cddn_min = min(cddn);

                p = patch('XData',[cddn_min cddn_max cddn_max cddn_min],'YData',[0 0 1 1],'EdgeAlpha',0.25,'FaceAlpha',0.40,'FaceColor',co(i,:));
                set(p,'Tag',num2str(i));
            end
        hold off;

        ax = gca();

        if (k <= 5)
            legend(ax,ds.CrisisNames,'Location','southoutside','Orientation','horizontal');

            tooltips = [];
            tooltips_target = [];
        else
            tooltips = ds.CrisisNames;
            tooltips_target = p;
        end
    end

    set(ax,'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
    set(ax,'YLim',[0 1],'YTick',[]);

    if (ds.MonthlyTicks)
        date_ticks(ax,'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        date_ticks(ax,'x','yyyy','KeepLimits');
    end

    if (strcmp(ds.CrisesType,'E'))
        figure_title('Crises (Events)');
    else
        figure_title('Crises (Ranges)');
    end

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

    if (~isempty(tooltips))
        drawnow();

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

    index = ds.Index;

    index_obs = numel(index);
    index_max = max(index);
    index_min = min(index);

    index_avg = mean(index);
    index_med = median(index);
    index_std = std(index);
    index_ske = skewness(index,0);
    index_kur = kurtosis(index,0);

    f = figure('Name','Dataset > Index','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    sub_1 = subplot(2,1,1);
    plot(sub_1,ds.DatesNum,ds.Index);
    set(sub_1,'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
    set(sub_1,'YLim',[(index_min - 0.01) (index_max + 0.01)]);
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
    [values,points] = ksdensity(ds.Index);
    hold on;
        plot(sub_2,points,values,'-b','LineWidth',1.5);
    hold off;
    set(sub_2,'XLim',[(edges_min - (edges_min * 0.1)) (edges_max - (edges_max * 0.1))]);
    t2 = title(sub_2,'P&L Distribution');
    set(t2,'Units','normalized');
    t2_position = get(t2,'Position');
    set(t2,'Position',[0.4783 t2_position(2) t2_position(3)]);

    t = figure_title(['Index (' ds.IndexName ')']);
    t_position = get(t,'Position');
    set(t,'Position',[t_position(1) -0.0157 t_position(3)]);

    txt = {sprintf('Observations: %d',index_obs) sprintf('Mean: %.4f',index_avg) sprintf('Median: %.4f',index_med) sprintf('Standard Deviation: %.4f',index_std) sprintf('Skewness: %.4f',index_ske) sprintf('Kurtosis: %.4f',index_kur)};
    annotation('TextBox',(get(sub_2,'Position') + [0.01 -0.025 0 0]),'String',txt,'EdgeColor','none','FitBoxToText','on','FontSize',8);

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

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

    figure_title('Risk-Free Rate');

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

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

        y_obs = numel(y);
        y_max = max(y,[],'omitnan');
        y_min = min(y,[],'omitnan');

        y_avg = mean(y,'omitnan');
        y_med = median(y,'omitnan');
        y_std = std(y,'omitnan');
        y_ske = skewness(y,0);
        y_kur = kurtosis(y,0);

        d = find(isnan(y),1,'first');

        if (isempty(d))
            xd = [];
        else
            xd = x(d) - 1;
        end

        delete(findall(gcf,'type','annotation'));

        plot(subs(1),x,y,'Color',[0.000 0.447 0.741]);
        set(subs(1),'YLim',[(y_min - 0.01) (y_max + 0.01)]);

        if (~isempty(xd))
            hold(subs(1),'on');
                plot(subs(1),[xd xd],get(subs(1),'YLim'),'Color',[1 0.4 0.4]);
            hold(subs(1),'off');
        end

        hist = histogram(subs(2),y,50,'FaceColor',[0.749 0.862 0.933],'Normalization','pdf');
        edges = get(hist,'BinEdges');
        edges_max = max(edges);
        edges_min = min(edges);
        [values,points] = ksdensity(y);

        hold(subs(2),'on');
            plot(subs(2),points,values,'-b','LineWidth',1.5);
        hold(subs(2),'off');
        set(subs(2),'XLim',[(edges_min - (edges_min * 0.1)) (edges_max - (edges_max * 0.1))]);

        txt = {sprintf('Observations: %d',y_obs) sprintf('Mean: %.4f',y_avg) sprintf('Median: %.4f',y_med) sprintf('Standard Deviation: %.4f',y_std) sprintf('Skewness: %.4f',y_ske) sprintf('Kurtosis: %.4f',y_kur)};
        annotation('TextBox',(get(subs(2),'Position') + [0.01 -0.025 0 0]),'String',txt,'EdgeColor','none','FitBoxToText','on','FontSize',8);

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

        plot(subs(1),x,y,'Color',[0.000 0.447 0.741]);

        if (~isempty(xd))
            hold(subs(1),'on');
                plot(subs(1),[xd xd],get(subs(1),'YLim'),'Color',[1 0.4 0.4]);
            hold(subs(1),'off');
        end

    end

end
