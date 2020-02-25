% [INPUT]
% data = A structure representing the dataset.

function analyze_dataset(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data',@(x)validateattributes(x,{'struct'},{'nonempty'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    data = validate_dataset(ipr.data);
    
    nargoutchk(0,0);

    analyze_dataset_internal(data);

end

function analyze_dataset_internal(data)

    plot_index(data);
    plot_returns(data);
    
    if (~isempty(data.Capitalizations))
        plot_capitalizations(data);
    end
    
    if (~isempty(data.Assets) && ~isempty(data.Equity))
        plot_assets(data);
        plot_equity(data);
        plot_liabilities(data);
    end

end

function plot_index(data)

    f = figure('Name','Dataset > Index','Units','normalized','Position',[100 100 0.85 0.85]);

    sub_1 = subplot(2,1,1);
    plot(sub_1,data.DatesNum,data.IndexReturns);
    set(sub_1,'XLim',[data.DatesNum(1) data.DatesNum(end)],'YLim',[(min(data.IndexReturns) - 0.01) (max(data.IndexReturns) + 0.01)],'XTickLabelRotation',45);
    t1 = title(sub_1,'Log Returns');
    set(t1,'Units','normalized');
    t1_position = get(t1,'Position');
    set(t1,'Position',[0.4783 t1_position(2) t1_position(3)]);

    if (data.MonthlyTicks)
        datetick(sub_1,'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        datetick(sub_1,'x','yyyy','KeepLimits');
    end
    
    sub_2 = subplot(2,1,2);
    hist = histogram(sub_2,data.IndexReturns,50,'FaceColor',[0.749 0.862 0.933],'Normalization','pdf');
    edges = get(hist,'BinEdges');
    edges_max = max(edges);
    edges_min = min(edges);
    [values,points] = ksdensity(data.IndexReturns);
    hold on;
        plot(sub_2,points,values,'-b','LineWidth',1.5);
    hold off;
    set(sub_2,'XLim',[(edges_min - (edges_min * 0.1)) (edges_max - (edges_max * 0.1))]);
    t2 = title(sub_2,'P&L Distribution');
    set(t2,'Units','normalized');
    t2_position = get(t2,'Position');
    set(t2,'Position',[0.4783 t2_position(2) t2_position(3)]);

    t = figure_title(['Index (' data.IndexName ')']);
    t_position = get(t,'Position');
    set(t,'Position',[t_position(1) -0.0157 t_position(3)]);
    
    annotation_strings = {sprintf('Observations: %d',size(data.IndexReturns,1)) sprintf('Kurtosis: %.4f',kurtosis(data.IndexReturns)) sprintf('Mean: %.4f',mean(data.IndexReturns)) sprintf('Median: %.4f',median(data.IndexReturns)) sprintf('Skewness: %.4f',skewness(data.IndexReturns)) sprintf('Standard Deviation: %.4f',std(data.IndexReturns))};
    annotation('TextBox',(get(sub_2,'Position') + [0.01 -0.025 0 0]),'String',annotation_strings,'EdgeColor','none','FitBoxToText','on','FontSize',8);
    
    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_returns(data)

    f = figure('Name','Dataset > Returns','Units','normalized','Position',[100 100 0.85 0.85]);    

    boxplot(data.FirmReturns,'Notch','on','Symbol','k.');
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
    
    set(gca(),'TickLength',[0 0],'XTick',1:data.N,'XTickLabels',data.FirmNames,'XTickLabelRotation',45,'YLim',[y_low y_high]);

    t = figure_title('Returns');
    t_position = get(t,'Position');
    set(t,'Position',[t_position(1) -0.0157 t_position(3)]);

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_capitalizations(data)

    f = figure('Name','Dataset > Market Capitalization','Units','normalized','Position',[100 100 0.85 0.85]);    

    boxplot(data.Capitalizations,'Notch','on','Symbol','k.');
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
    
    set(gca(),'TickLength',[0 0],'XTick',1:data.N,'XTickLabels',data.FirmNames,'XTickLabelRotation',45,'YLim',[y_low y_high]);

    t = figure_title('Market Capitalization');
    t_position = get(t,'Position');
    set(t,'Position',[t_position(1) -0.0157 t_position(3)]);

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_assets(data)

    f = figure('Name','Dataset > Assets','Units','normalized','Position',[100 100 0.85 0.85]);    

    boxplot(data.Assets,'Notch','on','Symbol','k.');
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
    
    set(gca(),'TickLength',[0 0],'XTick',1:data.N,'XTickLabels',data.FirmNames,'XTickLabelRotation',45,'YLim',[y_low y_high]);

    t = figure_title('Assets');
    t_position = get(t,'Position');
    set(t,'Position',[t_position(1) -0.0157 t_position(3)]);

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_equity(data)

    f = figure('Name','Dataset > Equity','Units','normalized','Position',[100 100 0.85 0.85]);    

    boxplot(data.Equity,'Notch','on','Symbol','k.');
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
    
    set(gca(),'TickLength',[0 0],'XTick',1:data.N,'XTickLabels',data.FirmNames,'XTickLabelRotation',45,'YLim',[y_low y_high]);

    t = figure_title('Equity');
    t_position = get(t,'Position');
    set(t,'Position',[t_position(1) -0.0157 t_position(3)]);

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_liabilities(data)

    f = figure('Name','Dataset > Liabilities','Units','normalized','Position',[100 100 0.85 0.85]);    

    boxplot(data.Liabilities,'Notch','on','Symbol','k.');
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
    
    set(gca(),'TickLength',[0 0],'XTick',1:data.N,'XTickLabels',data.FirmNames,'XTickLabelRotation',45,'YLim',[y_low y_high]);

    t = figure_title('Liabilities');
    t_position = get(t,'Position');
    set(t,'Position',[t_position(1) -0.0157 t_position(3)]);

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end
