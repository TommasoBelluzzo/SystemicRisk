% [INPUT]
% ds = A structure representing the dataset.
% temp = A string representing the full path to the Excel spreadsheet used as a template for the results file.
% out = A string representing the full path to the Excel spreadsheet to which the results are written, eventually replacing the previous ones.
% bw = An integer [21,252] representing the dimension of each rolling window (optional, default=252).
% sst = A float (0.0,0.1] representing the statistical significance threshold for the linear Granger-causality test (optional, default=0.05).
% rp = A boolean indicating whether to use robust p-values for the linear Granger-causality test (optional, default=false).
% k = A float (0.00,0.20] representing the Granger-causality threshold for non-causal relationships (optional, default=0.06).
% analyze = A boolean that indicates whether to analyse the results and display plots (optional, default=false).
%
% [OUTPUT]
% result = A structure representing the original dataset inclusive of intermediate and final calculations.
% stopped = A boolean that indicates whether the process has been stopped through user input.

function [result,stopped] = run_connectedness(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('ds',@(x)validateattributes(x,{'struct'},{'nonempty'}));
        ip.addRequired('temp',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addRequired('out',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addOptional('bw',252,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 21 '<=' 252 'scalar'}));
        ip.addOptional('sst',0.05,@(x)validateattributes(x,{'double'},{'real' 'finite' '>' 0 '<=' 0.1 'scalar'}));
        ip.addOptional('rp',false,@(x)validateattributes(x,{'logical'},{'scalar'}));
        ip.addOptional('k',0.06,@(x)validateattributes(x,{'double'},{'real' 'finite' '>' 0 '<=' 0.20 'scalar'}));
        ip.addOptional('analyze',false,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    ds = validate_dataset(ipr.ds,'connectedness');
    temp = validate_template(ipr.temp);
    out = validate_output(ipr.out);
    bw = ipr.bw;
    sst = ipr.sst;
    rp = ipr.rp;
    k = ipr.k;
    analyze = ipr.analyze;
    
    nargoutchk(1,2);
    
    [result,stopped] = run_connectedness_internal(ds,temp,out,bw,sst,rp,k,analyze);

end

function [result,stopped] = run_connectedness_internal(ds,temp,out,bw,sst,rp,k,analyze)

    result = [];
    stopped = false;
    e = [];
    
    ds = initialize(ds,bw,sst,rp,k);
    t = ds.T;
    
    bar = waitbar(0,'Initializing connectedness measures...','CreateCancelBtn',@(src,event)setappdata(gcbf(),'Stop',true));
    setappdata(bar,'Stop',false);
    cleanup = onCleanup(@()delete(bar));
    
    pause(1);
    waitbar(0,bar,'Calculating connectedness measures...');
    pause(1);

    try

        windows = extract_rolling_windows(ds.Returns,ds.BW);

        futures(1:t) = parallel.FevalFuture;
        futures_max = 0;
        futures_results = cell(t,1);

        for i = 1:t
            futures(i) = parfeval(@main_loop,1,windows{i},ds.SST,ds.RP,ds.GroupDelimiters);
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
    waitbar(1,bar,'Finalizing connectedness measures...');
    pause(1);

    try
        ds = finalize(ds,futures_results);
    catch e
        delete(bar);
        rethrow(e);
    end

    pause(1);
    waitbar(1,bar,'Writing connectedness measures...');
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
        safe_plot(@(id)plot_network(ds,id));
        safe_plot(@(id)plot_adjacency_matrix(ds,id));
        safe_plot(@(id)plot_centralities(ds,id));
    end
    
    result = ds;

end

%% DATA

function ds = initialize(ds,bw,sst,rp,k)

    n = ds.N;
    t = ds.T;

    ds.BW = bw;
    ds.K = k;
    ds.RP = rp;
    ds.SST = sst;

    if (ds.RP)
        all_label = [' (SST=' num2str(ds.SST) ', R)'];
    else
        all_label = [' (SST=' num2str(ds.SST) ')'];
    end
    
    ds.LabelsCentralities = {'Betweenness Centrality' 'Closeness Centrality' 'Degree Centrality' 'Eigenvector Centrality' 'Katz Centrality' 'Clustering Coefficient'};

    ds.LabelsIndicatorsSimple = {'DCI' 'CIO' 'CIOO'};
    ds.LabelsIndicatorsShort = ds.LabelsIndicatorsSimple;
    ds.LabelsIndicators = {['DCI' all_label] ['CIO' all_label] ['CIOO' all_label]};

    ds.LabelsSheetsSimple = {'Indicators' 'Average Adjacency Matrix' 'Average Centrality Measures'};
    ds.LabelsSheets = {['Indicators' all_label] 'Average Adjacency Matrix' 'Average Centrality Measures'};
    
    ds.AdjacencyMatrices = cell(t,1);
    ds.BetweennessCentralities = NaN(t,n);
    ds.ClosenessCentralities = NaN(t,n);
    ds.DegreeCentralities = NaN(t,n);
    ds.EigenvectorCentralities = NaN(t,n);
    ds.KatzCentralities = NaN(t,n);
    ds.ClusteringCoefficients = NaN(t,n);
    ds.Degrees = NaN(t,n);
    ds.DegreesIn = NaN(t,n);
    ds.DegreesOut = NaN(t,n);
    
    ds.Indicators = NaN(t,numel(ds.LabelsIndicators));

    ds.AverageAdjacencyMatrix = NaN(n);
    ds.AverageBetweennessCentralities = NaN(1,n);
    ds.AverageClosenessCentralities = NaN(1,n);
    ds.AverageDegreeCentralities = NaN(1,n);
    ds.AverageEigenvectorCentralities = NaN(1,n);
    ds.AverageKatzCentralities = NaN(1,n);
    ds.AverageClusteringCoefficients = NaN(1,n);
    ds.AverageDegreesIn = NaN(1,n);
    ds.AverageDegreesOut = NaN(1,n);
    ds.AverageDegrees = NaN(1,n);
    
    if (ds.Groups == 0)
        ds.ComparisonReferences = {'Indicators' 1:2 strcat({'CO-'},ds.LabelsIndicatorsShort)};
    else
        ds.ComparisonReferences = {'Indicators' 1:3 strcat({'CO-'},ds.LabelsIndicatorsShort)};
    end

end

function ds = finalize(ds,results)

    t = ds.T;

    for i = 1:t
        result = results{i};

        ds.AdjacencyMatrices{i} = result.AdjacencyMatrix;
        ds.BetweennessCentralities(i,:) = result.BetweennessCentralities;
        ds.ClosenessCentralities(i,:) = result.ClosenessCentralities;
        ds.DegreeCentralities(i,:) = result.DegreeCentralities;
        ds.EigenvectorCentralities(i,:) = result.EigenvectorCentralities;
        ds.KatzCentralities(i,:) = result.KatzCentralities;
        ds.ClusteringCoefficients(i,:) = result.ClusteringCoefficients;
        ds.Degrees(i,:) = result.Degrees;
        ds.DegreesIn(i,:) = result.DegreesIn;
        ds.DegreesOut(i,:) = result.DegreesOut;
        
        ds.Indicators(i,:) = [result.DCI result.ConnectionsInOut result.ConnectionsInOutOther];
    end

    am = sum(cat(3,ds.AdjacencyMatrices{:}),3) ./ numel(ds.AdjacencyMatrices);
    am_threshold = mean(mean(am));
    am(am < am_threshold) = 0;
    am(am >= am_threshold) = 1;
    ds.AverageAdjacencyMatrix = am;
    
    [bc,cc,dc,ec,kc,clc,deg,deg_in,deg_out] = network_centralities(am);
    ds.AverageBetweennessCentralities = bc;
    ds.AverageClosenessCentralities = cc;
    ds.AverageDegreeCentralities = dc;
    ds.AverageEigenvectorCentralities = ec;
    ds.AverageKatzCentralities = kc;
    ds.AverageClusteringCoefficients = clc;
    ds.AverageDegrees = deg;
    ds.AverageDegreesIn = deg_in;
    ds.AverageDegreesOut = deg_out;

end

function out = validate_output(out)

    [path,name,extension] = fileparts(out);

    if (~strcmp(extension,'.xlsx'))
        out = fullfile(path,[name extension '.xlsx']);
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
    
    sheets = {'Indicators' 'Average Adjacency Matrix' 'Average Centrality Measures'};

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
        error('The output file could not be created from the template file.');
    end

    firm_names = ds.FirmNames';

    vars = [ds.DatesStr num2cell(ds.Indicators)];
    labels = [{'Date'} ds.LabelsIndicatorsSimple];
    tab = cell2table(vars,'VariableNames',labels);
    writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{1},'WriteRowNames',true);

    vars = [firm_names num2cell(ds.AverageAdjacencyMatrix)];
    labels = {'Firms' ds.FirmNames{:,:}};
    tab = cell2table(vars,'VariableNames',labels);
    writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{2},'WriteRowNames',true);

    vars = [firm_names num2cell(ds.AverageBetweennessCentralities') num2cell(ds.AverageClosenessCentralities') num2cell(ds.AverageDegreeCentralities') num2cell(ds.AverageEigenvectorCentralities') num2cell(ds.AverageKatzCentralities') num2cell(ds.AverageClusteringCoefficients')];
    labels = [{'Firms'} strrep(ds.LabelsCentralities,' ','')];
    tab = cell2table(vars,'VariableNames',labels);
    writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{3},'WriteRowNames',true);
    
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

%% MEASURES

function window_results = main_loop(r,sst,rp,gd)

    window_results = struct();

    am = causal_adjacency(r,sst,rp);
    window_results.AdjacencyMatrix = am;

    [dci,cio,cioo] = calculate_connectedness_indicators(am,gd);
    window_results.DCI = dci;
    window_results.ConnectionsInOut = cio;
    window_results.ConnectionsInOutOther = cioo;

    [bc,cc,dc,ec,kc,clc,deg,deg_in,deg_out] = network_centralities(am);
    window_results.BetweennessCentralities = bc;
    window_results.ClosenessCentralities = cc;
    window_results.DegreeCentralities = dc;
    window_results.EigenvectorCentralities = ec;
    window_results.KatzCentralities = kc;
    window_results.ClusteringCoefficients = clc;
    window_results.Degrees = deg;
    window_results.DegreesIn = deg_in;
    window_results.DegreesOut = deg_out;

end

function [dci,cio,cioo] = calculate_connectedness_indicators(am,gd)

    n = size(am,1);

    dci = sum(sum(am)) / ((n ^ 2) - n);

    ni = zeros(n,1);
    no = zeros(n,1);
    
    for i = 1:n     
        ni(i) = sum(am(:,i));
        no(i) = sum(am(i,:));
    end

    cio = (sum(ni) + sum(no)) / (2 * (n - 1));
    
    if (isempty(gd))
        cioo = NaN;
    else
        gd_len = length(gd);

        nifo = zeros(n,1);
        noto = zeros(n,1);
        
        for i = 1:n
            group_1 = gd(1);
            group_n = gd(gd_len);
            
            if (i <= group_1)
                g_beg = 1;
                g_end = group_1;
            elseif (i > group_n)
                g_beg = group_n + 1;
                g_end = n;
            else
                for j = 1:gd_len-1
                    g_j0 = gd(j);
                    g_j1 = gd(j+1);

                    if ((i > g_j0) && (i <= g_j1))
                        g_beg = g_j0 + 1;
                        g_end = g_j1;
                    end
                end
            end

            nifo(i) = ni(i) - sum(am(g_beg:g_end,i));
            noto(i) = no(i) - sum(am(i,g_beg:g_end));
        end

        cioo = (sum(nifo) + sum(noto)) / (2 * gd_len * (n / gd_len));
    end

end

%% PLOTTING

function plot_indicators(ds,id)

    dci = smooth_data(ds.Indicators(:,1));
    cio = smooth_data(ds.Indicators(:,2));
    cioo = smooth_data(ds.Indicators(:,3));

    connections_max = max(max([cio cioo])) * 1.1;
    
    threshold_indices = dci >= ds.K;
    threshold = NaN(ds.T,1);
    threshold(threshold_indices) = connections_max;

    f = figure('Name','Connectedness Measures > Indicators','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);
    
    sub_1 = subplot(2,1,1);
    plot(sub_1,ds.DatesNum,dci);
    hold on;
        plot(sub_1,ds.DatesNum,repmat(ds.K,[ds.T 1]),'Color',[1 0.4 0.4]);
    hold off;
    set(sub_1,'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
    set(sub_1,'XGrid','on','YGrid','on');
    t1 = title(sub_1,ds.LabelsIndicators{1});
    set(t1,'Units','normalized');
    t1_position = get(t1,'Position');
    set(t1,'Position',[0.4783 t1_position(2) t1_position(3)]);

    sub_2 = subplot(2,1,2);
    a1 = area(sub_2,ds.DatesNum,threshold,'EdgeColor','none','FaceColor',[1 0.4 0.4]);
    hold on;
        a2 = area(sub_2,ds.DatesNum,cio,'EdgeColor','none','FaceColor','b');
        if (ds.Groups == 0)
            a3 = area(sub_2,ds.DatesNum,NaN(ds.T,1),'EdgeColor','none','FaceColor',[0.678 0.922 1]);
        else
            a3 = area(sub_2,ds.DatesNum,cioo,'EdgeColor','none','FaceColor',[0.678 0.922 1]);
        end
    hold off;
    set(sub_2,'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45,'YLim',[0 connections_max]);
    legend(sub_2,[a2 a3 a1],'In & Out','In & Out - Other','Granger-causality Threshold','Location','best');
    t2 = title(sub_2,strrep(ds.LabelsIndicators{2},'CIO','Connections'));
    set(t2,'Units','normalized');
    t2_position = get(t2,'Position');
    set(t2,'Position',[0.4783 t2_position(2) t2_position(3)]);

    if (ds.MonthlyTicks)
        date_ticks([sub_1 sub_2],'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        date_ticks([sub_1 sub_2],'x','yyyy','KeepLimits');
    end

    figure_title('Indicators');
    
    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_network(ds,id)

    if (ds.Groups == 0)
        group_colors = repmat(lines(1),ds.N,1);
    else
        group_colors = zeros(ds.N,3);
        group_delimiters_len = length(ds.GroupDelimiters);
        group_lines = lines(ds.Groups);

        for i = 1:group_delimiters_len
            group_delimiter = ds.GroupDelimiters(i);

            if (i == 1)
                group_colors(1:group_delimiter,:) = repmat(group_lines(i,:),group_delimiter,1);
            else
                group_delimiter_prev = ds.GroupDelimiters(i-1) + 1;
                group_colors(group_delimiter_prev:group_delimiter,:) = repmat(group_lines(i,:),group_delimiter - group_delimiter_prev + 1,1);
            end

            if (i == group_delimiters_len)
                group_colors(group_delimiter+1:end,:) = repmat(group_lines(i+1,:),ds.N - group_delimiter,1);
            end
        end
    end
    
    weights = mean(ds.Degrees,1,'omitnan');
    weights = weights ./ mean(weights);
    weights = (weights - min(weights)) ./ (max(weights) - min(weights));
    weights = (weights .* 3.75) + 0.25;
    
    theta = linspace(0,(2 * pi),(ds.N + 1)).';
    theta(end) = [];
    xy = [cos(theta) sin(theta)];
    [i,j] = find(ds.AverageAdjacencyMatrix);
    [~,order] = sort(max(i,j));
    i = i(order);
    j = j(order);
    x = [xy(i,1) xy(j,1)].';
    y = [xy(i,2) xy(j,2)].';

    f = figure('Name','Connectedness Measures > Network Graph','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    sub = subplot(100,1,10:100);

    hold on;
        for i = 1:size(x,2)
            index = ismember(xy,[x(1,i) y(1,i)],'rows');
            plot(sub,x(:,i),y(:,i),'Color',group_colors(index,:));
        end
    hold off;

    if (ds.Groups == 0)
        hold on;
            for i = 1:size(xy,1)
                line(xy(i,1),xy(i,2),'Color',group_colors(i,:),'LineStyle','none','Marker','.','MarkerSize',(35 + (15 * weights(i))));
            end
        hold off;
    else
        d_inc = ds.GroupDelimiters + 1;

        lines_ref = NaN(ds.Groups,1);
        lines_off = 1;

        hold on;
            for i = 1:size(xy,1)
                group_color = group_colors(i,:);
                line(xy(i,1),xy(i,2),'Color',group_color,'LineStyle','none','Marker','.','MarkerSize',(35 + (15 * weights(i))));

                if ((i == 1) || any(d_inc == i))
                    lines_ref(lines_off) = line(xy(i,1),xy(i,2),'Color',group_color,'LineStyle','none','Marker','.','MarkerSize',35);
                    lines_off = lines_off + 1;
                end
            end
        hold off;

        legend(sub,lines_ref,ds.GroupShortNames,'Units','normalized','Position',[0.181 0.716 0.040 0.076]);
    end

    axis(sub,[-1 1 -1 1]);
    axis('equal','off');

    labels = text((xy(:,1) .* 1.075), (xy(:,2) .* 1.075),ds.FirmNames,'FontSize',10);
    set(labels,{'Rotation'},num2cell(theta * (180 / pi())));

    figure_title('Network Graph');
    
    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_adjacency_matrix(ds,id)

    am = ds.AverageAdjacencyMatrix;
    am(logical(eye(ds.N))) = 0.5;
    am = padarray(am,[1 1],'post');

    off = ds.N + 0.5;

    f = figure('Name','Connectedness Measures > Average Adjacency Matrix','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    pcolor(am);
    colormap([1 1 1; 0.65 0.65 0.65; 0.749 0.862 0.933]);
    axis image;

    ax = gca();
    set(ax,'TickLength',[0 0]);
    set(ax,'XAxisLocation','top','XTick',1.5:off,'XTickLabels',ds.FirmNames,'XTickLabelRotation',45);
    set(ax,'YDir','reverse','YTick',1.5:off,'YTickLabels',ds.FirmNames,'YTickLabelRotation',45);
    
    figure_title('Average Adjacency Matrix');

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_centralities(ds,id)

    seq = 1:ds.N;
    
    [bc,order] = sort(ds.AverageBetweennessCentralities);
    bc_names = ds.FirmNames(order);
    [cc,order] = sort(ds.AverageClosenessCentralities);
    cc_names = ds.FirmNames(order);
    [dc,order] = sort(ds.AverageDegreeCentralities);
    dc_names = ds.FirmNames(order);
    [ec,order] = sort(ds.AverageEigenvectorCentralities);
    ec_names = ds.FirmNames(order);
    [kc,order] = sort(ds.AverageKatzCentralities);
    kc_names = ds.FirmNames(order);
    [clc,order] = sort(ds.AverageClusteringCoefficients);
    clc_names = ds.FirmNames(order);

    f = figure('Name','Connectedness Measures > Average Centrality Measures','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    sub_1 = subplot(2,3,1);
    bar(sub_1,seq,bc,'FaceColor',[0.749 0.862 0.933]);
    set(sub_1,'XTickLabel',bc_names);
    title(ds.LabelsCentralities{1});
    
    sub_2 = subplot(2,3,2);
    bar(sub_2,seq,cc,'FaceColor',[0.749 0.862 0.933]);
    set(sub_2,'XTickLabel',cc_names);
    title(ds.LabelsCentralities{2});
    
    sub_3 = subplot(2,3,3);
    bar(sub_3,seq,dc,'FaceColor',[0.749 0.862 0.933]);
    set(sub_3,'XTickLabel',dc_names);
    title(ds.LabelsCentralities{3});
    
    sub_4 = subplot(2,3,4);
    bar(sub_4,seq,ec,'FaceColor',[0.749 0.862 0.933]);
    set(sub_4,'XTickLabel',ec_names);
    title(ds.LabelsCentralities{4});
    
    sub_5 = subplot(2,3,5);
    bar(sub_5,seq,kc,'FaceColor',[0.749 0.862 0.933]);
    set(sub_5,'XTickLabel',kc_names);
    title(ds.LabelsCentralities{5});

    sub_6 = subplot(2,3,6);
    bar(sub_6,seq,clc,'FaceColor',[0.749 0.862 0.933]);
    set(sub_6,'XTickLabel',clc_names);
    title(ds.LabelsCentralities{6});
    
    set([sub_1 sub_2 sub_3 sub_4 sub_5 sub_6],'XLim',[0 (ds.N + 1)],'XTick',seq,'XTickLabelRotation',90);
    set([sub_1 sub_2 sub_3 sub_4 sub_5 sub_6],'YGrid','on');

    figure_title('Average Centrality Measures');
    
    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end
