% [INPUT]
% data = A structure representing the dataset.
% temp = A string representing the full path to the Excel spreadsheet used as a template for the results file.
% out = A string representing the full path to the Excel spreadsheet to which the results are written, eventually replacing the previous ones.
% bandwidth = An integer [21,252] representing the dimension of each rolling window (optional, default=252).
% significance = A float (0.0,0.1] representing the statistical significance threshold for the linear Granger-causality test (optional, default=0.05).
% robust = A boolean indicating whether to use robust p-values for the linear Granger-causality test (optional, default=false).
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
        ip.addRequired('data',@(x)validateattributes(x,{'struct'},{'nonempty'}));
        ip.addRequired('temp',@(x)validateattributes(x,{'char'},{'nonempty','size',[1 NaN]}));
        ip.addRequired('out',@(x)validateattributes(x,{'char'},{'nonempty','size',[1 NaN]}));
        ip.addOptional('bandwidth',252,@(x)validateattributes(x,{'double'},{'real','finite','integer','>=',21,'<=',252,'scalar'}));
        ip.addOptional('significance',0.05,@(x)validateattributes(x,{'double'},{'real','finite','>',0,'<=',0.1,'scalar'}));
        ip.addOptional('robust',false,@(x)validateattributes(x,{'logical'},{'scalar'}));
        ip.addOptional('k',0.06,@(x)validateattributes(x,{'double'},{'real','finite','>',0,'<=',0.20,'scalar'}));
        ip.addOptional('analyze',false,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    data = validate_dataset(ipr.data,'connectedness');
    temp = validate_template(ipr.temp);
    out = validate_output(ipr.out);
    
    nargoutchk(1,2);
    
    [result,stopped] = run_connectedness_internal(data,temp,out,ipr.bandwidth,ipr.significance,ipr.robust,ipr.k,ipr.analyze);

end

function [result,stopped] = run_connectedness_internal(data,temp,out,bandwidth,significance,robust,k,analyze)

    result = [];
    stopped = false;
    e = [];
    
    data = data_initialize(data,bandwidth,significance,robust,k);
    t = data.T;
    
    bar = waitbar(0,'Initializing connectedness measures...','CreateCancelBtn',@(src,event)setappdata(gcbf(),'Stop',true));
    setappdata(bar,'Stop',false);
    cleanup = onCleanup(@()delete(bar));
    
    pause(1);
    waitbar(0,bar,'Calculating connectedness measures...');
    pause(1);

    try

        r = data.Returns;
        r(isnan(r)) = 0;
        
        windows = extract_rolling_windows(r,bandwidth,false);

        futures(1:t) = parallel.FevalFuture;
        futures_max = 0;
        futures_results = cell(t,1);

        for i = 1:t
            futures(i) = parfeval(@main_loop,1,windows{i},data.Significance,data.Robust,data.GroupDelimiters);
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
        data = data_finalize(data,futures_results);
    catch e
        delete(bar);
        rethrow(e);
    end

    pause(1);
    waitbar(1,bar,'Writing connectedness measures...');
    pause(1);
    
    try
        write_results(temp,out,data);
        delete(bar);
    catch e
        delete(bar);
        rethrow(e);
    end

    if (analyze)
        safe_plot(@(id)plot_indicators(data,id));
        safe_plot(@(id)plot_network(data,id));
        safe_plot(@(id)plot_adjacency_matrix(data,id));
        safe_plot(@(id)plot_centralities(data,id));
    end
    
    result = data;

end

%% DATA

function data = data_initialize(data,bandwidth,significance,robust,k)

    data.Bandwidth = bandwidth;
    data.K = k;
    data.Robust = robust;
    data.Significance = significance;
    
    data.AdjacencyMatrices = cell(data.T,1);
    data.DCI = NaN(data.T,1);
    data.ConnectionsInOut = NaN(data.T,1);
    data.ConnectionsInOutOther = NaN(data.T,1);
    data.BetweennessCentralities = NaN(data.T,data.N);
    data.ClosenessCentralities = NaN(data.T,data.N);
    data.DegreeCentralities = NaN(data.T,data.N);
    data.EigenvectorCentralities = NaN(data.T,data.N);
    data.KatzCentralities = NaN(data.T,data.N);
    data.ClusteringCoefficients = NaN(data.T,data.N);
    data.DegreesIn = NaN(data.T,data.N);
    data.DegreesOut = NaN(data.T,data.N);
    data.Degrees = NaN(data.T,data.N);

    data.AdjacencyMatrixAverage = [];
    data.BetweennessCentralitiesAverage = [];
    data.ClosenessCentralitiesAverage = [];
    data.DegreeCentralitiesAverage = [];
    data.EigenvectorCentralitiesAverage = [];
    data.KatzCentralitiesAverage = [];
    data.ClusteringCoefficientsAverage = [];
    data.DegreesInAverage = [];
    data.DegreesOutAverage = [];
    data.DegreesAverage = [];

end

function data = data_finalize(data,window_results)

    t = data.T;

    for i = 1:t
        futures_result = window_results{i};

        data.AdjacencyMatrices{i} = futures_result.AdjacencyMatrix;
        data.DCI(i) = futures_result.DCI;
        data.ConnectionsInOut(i) = futures_result.ConnectionsInOut;
        data.ConnectionsInOutOther(i) = futures_result.ConnectionsInOutOther;
        data.BetweennessCentralities(i,:) = futures_result.BetweennessCentralities;
        data.ClosenessCentralities(i,:) = futures_result.ClosenessCentralities;
        data.DegreeCentralities(i,:) = futures_result.DegreeCentralities;
        data.EigenvectorCentralities(i,:) = futures_result.EigenvectorCentralities;
        data.KatzCentralities(i,:) = futures_result.KatzCentralities;
        data.ClusteringCoefficients(i,:) = futures_result.ClusteringCoefficients;
        data.DegreesIn(i,:) = futures_result.DegreesIn;
        data.DegreesOut(i,:) = futures_result.DegreesOut;
        data.Degrees(i,:) = futures_result.Degrees;
    end

    am_avg = sum(cat(3,data.AdjacencyMatrices{:}),3) ./ t;
    am_threshold = mean(mean(am_avg));
    am_avg(am_avg < am_threshold) = 0;
    am_avg(am_avg >= am_threshold) = 1;
    data.AdjacencyMatrixAverage = am_avg;
    
    [bc,cc,dc,ec,kc,clc,deg_in,deg_out,deg] = calculate_centralities(am_avg);
    data.BetweennessCentralitiesAverage = bc;
    data.ClosenessCentralitiesAverage = cc;
    data.DegreeCentralitiesAverage = dc;
    data.EigenvectorCentralitiesAverage = ec;
    data.KatzCentralitiesAverage = kc;
    data.ClusteringCoefficientsAverage = clc;
    data.DegreesInAverage = deg_in;
    data.DegreesOutAverage = deg_out;
    data.DegreesAverage = deg;

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

function write_results(temp,out,data)

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

    firm_names = data.FirmNames';

    vars = [data.DatesStr num2cell(data.DCI) num2cell(data.ConnectionsInOut) num2cell(data.ConnectionsInOutOther)];
    labels = {'Date' 'DCI' 'Connections_InOut' 'Connections_InOutOther'};
    t1 = cell2table(vars,'VariableNames',labels);
    writetable(t1,out,'FileType','spreadsheet','Sheet','Indicators','WriteRowNames',true);

    vars = [firm_names num2cell(data.AdjacencyMatrixAverage)];
    labels = {'Firms' data.FirmNames{:,:}};
    t2 = cell2table(vars,'VariableNames',labels);
    writetable(t2,out,'FileType','spreadsheet','Sheet','Average Adjacency Matrix','WriteRowNames',true);

    vars = [firm_names num2cell(data.BetweennessCentralitiesAverage') num2cell(data.ClosenessCentralitiesAverage') num2cell(data.DegreeCentralitiesAverage') num2cell(data.EigenvectorCentralitiesAverage') num2cell(data.KatzCentralitiesAverage') num2cell(data.ClusteringCoefficientsAverage')];
    labels = {'Firms' 'BetweennessCentrality' 'ClosenessCentrality' 'DegreeCentrality' 'EigenvectorCentrality' 'KatzCentrality' 'ClusteringCoefficient'};
    t3 = cell2table(vars,'VariableNames',labels);
    writetable(t3,out,'FileType','spreadsheet','Sheet','Average Centrality Measures','WriteRowNames',true);
    
end

%% MEASURES

function window_results = main_loop(window,significance,robust,group_delimiters)

    window_results = struct();

    am = causal_adjacency(window,significance,robust);
    window_results.AdjacencyMatrix = am;

    [dci,number_io,number_ioo] = calculate_connectedness_indicators(am,group_delimiters);
    window_results.DCI = dci;
    window_results.ConnectionsInOut = number_io;
    window_results.ConnectionsInOutOther = number_ioo;

    [bc,cc,dc,ec,kc,clc,deg_in,deg_out,deg] = calculate_centralities(am);
    window_results.BetweennessCentralities = bc;
    window_results.ClosenessCentralities = cc;
    window_results.DegreeCentralities = dc;
    window_results.EigenvectorCentralities = ec;
    window_results.KatzCentralities = kc;
    window_results.ClusteringCoefficients = clc;
    window_results.DegreesIn = deg_in;
    window_results.DegreesOut = deg_out;
    window_results.Degrees = deg;

end

function [bc,cc,dc,ec,kc,clc,deg_in,deg_out,deg] = calculate_centralities(am)

    am_len = length(am);

    bc = calculate_betweenness_centrality(am,am_len);
    [deg_in,deg_out,deg,dc] = calculate_degree_centrality(am,am_len);
    cc = calculate_closeness_centrality(am,am_len);
    ec = calculate_eigenvector_centrality(am);
    kc = calculate_katz_centrality(am,am_len);
    clc = calculate_clustering_coefficient(am,am_len,deg);

end

function [dci,in_out,in_out_other] = calculate_connectedness_indicators(am,group_delimiters)

    n = size(am,1);

    dci = sum(sum(am)) / ((n ^ 2) - n);

    number_i = zeros(n,1);
    number_o = zeros(n,1);
    
    for i = 1:n     
        number_i(i) = sum(am(:,i));
        number_o(i) = sum(am(i,:));
    end

    in_out = (sum(number_i) + sum(number_o)) / (2 * (n - 1));
    
    if (isempty(group_delimiters))
        in_out_other = NaN;
    else
        group_delimiters_len = length(group_delimiters);
        number_ifo = zeros(n,1);
        number_oto = zeros(n,1);
        
        for i = 1:n
            group_1 = group_delimiters(1);
            group_n = group_delimiters(group_delimiters_len);
            
            if (i <= group_1)
                group_begin = 1;
                group_end = group_1;
            elseif (i > group_n)
                group_begin = group_n + 1;
                group_end = n;
            else
                for j = 1:group_delimiters_len-1
                    group_j0 = group_delimiters(j);
                    group_j1 = group_delimiters(j+1);

                    if ((i > group_j0) && (i <= group_j1))
                        group_begin = group_j0 + 1;
                        group_end = group_j1;
                    end
                end
            end

            number_ifo(i) = number_i(i) - sum(am(group_begin:group_end,i));
            number_oto(i) = number_o(i) - sum(am(i,group_begin:group_end));
        end

        in_out_other = (sum(number_ifo) + sum(number_oto)) / (2 * group_delimiters_len * (n / group_delimiters_len));
    end

end

function bc = calculate_betweenness_centrality(am,am_len)

    bc = zeros(1,am_len);

    for i = 1:am_len
        depth = 0;
        nsp = accumarray([1 i],1,[1 am_len]);
        bfs = false(250,am_len);
        fringe = am(i,:);

        while ((nnz(fringe) > 0) && (depth <= 250))
            depth = depth + 1;
            nsp = nsp + fringe;
            bfs(depth,:) = logical(fringe);
            fringe = (fringe * am) .* ~nsp;
        end

        [rows,cols,v] = find(nsp);
        v = 1 ./ v;
        
        nsp_inv = accumarray([rows.' cols.'],v,[1 am_len]);

        bcu = ones(1,am_len);

        for depth = depth:-1:2
            w = (bfs(depth,:) .* nsp_inv) .* bcu;
            bcu = bcu + ((am * w.').' .* bfs(depth-1,:)) .* nsp;
        end

        bc = bc + sum(bcu,1);
    end

    bc = bc - am_len;
    bc = (bc .* 2) ./ ((am_len - 1) * (am_len - 2));

end

function cc = calculate_closeness_centrality(am,am_len)

    cc = zeros(1,am_len);

    for i = 1:am_len
        paths = dijkstra_shortest_paths(am,am_len,i);
        paths_sum = sum(paths(~isinf(paths)));
        
        if (paths_sum ~= 0)
            cc(i) = 1 / paths_sum;
        end
    end

    cc = cc .* (am_len - 1);

end

function clc = calculate_clustering_coefficient(am,am_len,deg)

    if (issymmetric(am))
        f = 2;
    else
        f = 1;
    end

    clc = zeros(am_len,1);

    for i = 1:am_len
        degree = deg(i);

        if ((degree == 0) || (degree == 1))
            continue;
        end

        k_neighbors = find(am(i,:) ~= 0);
        k_subgraph = am(k_neighbors,k_neighbors);

        if (issymmetric(k_subgraph))
            k_subgraph_trace = trace(k_subgraph);
            
            if (k_subgraph_trace == 0)
                edges = sum(sum(k_subgraph)) / 2; 
            else
                edges = ((sum(sum(k_subgraph)) - k_subgraph_trace) / 2) + k_subgraph_trace;
            end
        else
            edges = sum(sum(k_subgraph));
        end

        clc(i) = (f * edges) / (degree * (degree - 1));     
    end
    
    clc = clc.';

end

function [deg_in,deg_out,deg,dc] = calculate_degree_centrality(am,am_len)

    deg_in = sum(am);
    deg_out = sum(am.');
    
    if (issymmetric(am))
        deg = deg_in + diag(am).';
    else
        deg = deg_in + deg_out;
    end

    dc = deg ./ (am_len - 1);

end

function ec = calculate_eigenvector_centrality(am)

	[eigen_vector,eigen_values] = eig(am);
    [~,indices] = max(diag(eigen_values));

    ec = abs(eigen_vector(:,indices)).';
    ec = ec ./ sum(ec);

end

function kc = calculate_katz_centrality(am,am_len)

    kc = (eye(am_len) - (am .* 0.1)) \ ones(am_len,1);
    kc = kc.' ./ (sign(sum(kc)) * norm(kc,'fro'));

end

function am = causal_adjacency(data,significance,robust)

    n = size(data,2);
    i = repelem((1:n).',n,1);
    j = repmat((1:n).',n,1); 

    indices = i == j;
    i(indices) = [];
    j(indices) = [];

    d_in = arrayfun(@(x)data(:,x),i,'UniformOutput',false);
    d_out = arrayfun(@(x)data(:,x),j,'UniformOutput',false);
    
    k = n^2 - n;
    pvalues = zeros(k,1);

    if (robust)
        for y = 1:k
            in = d_in{y};
            out = d_out{y};

            [~,pvalues(y)] = linear_granger_causality(in,out);
        end
    else
        for y = 1:k
            in = d_in{y};
            out = d_out{y};

            [pvalues(y),~] = linear_granger_causality(in,out);
        end
    end

    am = zeros(n);
    am(sub2ind([n n],i,j)) = pvalues < significance;

end

function paths = dijkstra_shortest_paths(am,am_len,node)

    paths = Inf(1,am_len);
    paths(node) = 0;

    s = 1:am_len;

    while (~isempty(s))
        [~,idx] = min(paths(s));
        s_min = s(idx);

        for i = 1:length(s)
            s_i = s(i);

            offset = am(s_min,s_i);
            offset_sum = offset + paths(s_min);
            
            if ((offset > 0) && (paths(s_i) > offset_sum))
                paths(s_i) = offset_sum;
            end
        end

        s = setdiff(s,s_min);
    end

end

function [beta,covariance,residuals] = hac_regression(y,x,ratio)

    t = length(y);

    [beta,~,residuals] = regress(y,x);

    h = diag(residuals) * x;
    q_hat = (x.' * x) / t;
    o_hat = (h.' * h) / t;
    
    l = round(ratio * t,0);
    
	for i = 1:(l - 1)
        o_tmp = (h(1:(t-i),:).' * h((1+i):t,:)) / (t - i);
        o_hat = o_hat + (((l - i) / l) * (o_tmp + o_tmp.'));
	end

    covariance = (q_hat \ o_hat) / q_hat;

end

function [pval,pval_robust] = linear_granger_causality(in,out)
    
    t = length(in);
    y = out(2:t,1);
    x = [out(1:t-1) in(1:t-1)];

    [beta,covariance,residuals] = hac_regression(y,x,0.1);

    c = inv(x.' * x);
    s2 = (residuals.' * residuals) / (t - 3);
    t_coefficients = beta(2) / sqrt(s2 * c(2,2));
    
    pval = 1 - normcdf(t_coefficients);
    pval_robust = 1 - normcdf(beta(2) / sqrt(covariance(2,2) / (t - 1)));

end

%% PLOTTING

function plot_indicators(data,id)

    connections_max = max(max([data.ConnectionsInOut data.ConnectionsInOutOther])) * 1.1;
    
    threshold_indices = data.DCI >= data.K;
    threshold = NaN(data.T,1);
    threshold(threshold_indices) = connections_max;

    f = figure('Name','Connectedness Measures > Indicators','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);
    
    sub_1 = subplot(2,1,1);
    plot(sub_1,data.DatesNum,data.DCI);
    hold on;
        plot(sub_1,data.DatesNum,repmat(data.K,[data.T 1]),'Color',[1 0.4 0.4]);
    hold off;
    t1 = title(sub_1,'Dynamic Causality Index');
    set(t1,'Units','normalized');
    t1_position = get(t1,'Position');
    set(t1,'Position',[0.4783 t1_position(2) t1_position(3)]);

    sub_2 = subplot(2,1,2);
    area_1 = area(sub_2,data.DatesNum,threshold,'EdgeColor','none','FaceColor',[1 0.4 0.4]);
    hold on;
        area_2 = area(sub_2,data.DatesNum,data.ConnectionsInOut,'EdgeColor','none','FaceColor','b');
        if (data.Groups == 0)
            area_3 = area(sub_2,data.DatesNum,NaN(data.T,1),'EdgeColor','none','FaceColor',[0.678 0.922 1]);
        else
            area_3 = area(sub_2,data.DatesNum,data.ConnectionsInOutOther,'EdgeColor','none','FaceColor',[0.678 0.922 1]);
        end
    hold off;
    set(sub_2,'YLim',[0 connections_max]);
    legend(sub_2,[area_2 area_3 area_1],'In & Out','In & Out - Other','Granger-causality Threshold','Location','best');
    t2 = title(sub_2,'Connections');
    set(t2,'Units','normalized');
    t2_position = get(t2,'Position');
    set(t2,'Position',[0.4783 t2_position(2) t2_position(3)]);

    if (data.MonthlyTicks)
        datetick(sub_1,'x','mm/yyyy','KeepLimits','KeepTicks');
        datetick(sub_2,'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        datetick(sub_1,'x','yyyy','KeepLimits');
        datetick(sub_2,'x','yyyy','KeepLimits');
    end
    
    set([sub_1 sub_2],'XLim',[data.DatesNum(1) data.DatesNum(end)],'XTickLabelRotation',45);
    
    t = figure_title('Indicators');
    t_position = get(t,'Position');
    set(t,'Position',[t_position(1) -0.0157 t_position(3)]);
    
    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_network(data,id)

    if (data.Groups == 0)
        group_colors = repmat(lines(1),data.N,1);
    else
        group_colors = zeros(data.N,3);
        group_delimiters_len = length(data.GroupDelimiters);
        group_lines = lines(data.Groups);

        for i = 1:group_delimiters_len
            group_delimiter = data.GroupDelimiters(i);

            if (i == 1)
                group_colors(1:group_delimiter,:) = repmat(group_lines(i,:),group_delimiter,1);
            else
                group_delimiter_prev = data.GroupDelimiters(i-1) + 1;
                group_colors(group_delimiter_prev:group_delimiter,:) = repmat(group_lines(i,:),group_delimiter - group_delimiter_prev + 1,1);
            end

            if (i == group_delimiters_len)
                group_colors(group_delimiter+1:end,:) = repmat(group_lines(i+1,:),data.N - group_delimiter,1);
            end
        end
    end
    
    if (isempty(data.Capitalization))
        weights = mean(data.Degrees,1,'omitnan');
        weights = weights ./ mean(weights);
    else
        weights = mean(data.Capitalization,1,'omitnan');
        weights = weights ./ mean(weights);
    end

	weights_min = min(weights);
	weights = (weights - weights_min) ./ (max(weights) - min(weights));
    weights = (weights .* 3.75) + 0.25;
    
    theta = linspace(0,(2 * pi),(data.N + 1)).';
    theta(end) = [];
    xy = [cos(theta) sin(theta)];
    [i,j] = find(data.AdjacencyMatrixAverage);
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

    if (data.Groups == 0)
        hold on;
            for i = 1:size(xy,1)
                line(xy(i,1),xy(i,2),'Color',group_colors(i,:),'LineStyle','none','Marker','.','MarkerSize',(35 + (15 * weights(i))));
            end
        hold off;
    else
        d_inc = data.GroupDelimiters + 1;

        lines_ref = NaN(data.Groups,1);
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

        legend(sub,lines_ref,data.GroupNames,'Units','normalized','Position',[0.85 0.12 0.001 0.001]);
    end

    axis(sub,[-1 1 -1 1]);
    axis equal off;

    labels = text((xy(:,1) .* 1.075), (xy(:,2) .* 1.075),data.FirmNames,'FontSize',10);
    set(labels,{'Rotation'},num2cell(theta * (180 / pi())));

    t = figure_title('Network Graph');
    t_position = get(t,'Position');
    set(t,'Position',[t_position(1) -0.0157 t_position(3)]);
    
    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_adjacency_matrix(data,id)

    am = data.AdjacencyMatrixAverage;
    am(logical(eye(data.N))) = 0.5;
    am = padarray(am,[1 1],'post');

    off = data.N + 0.5;

    f = figure('Name','Connectedness Measures > Average Adjacency Matrix','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    pcolor(am);
    colormap([1 1 1; 0.65 0.65 0.65; 0.749 0.862 0.933])
    axis image;

    ax = gca();
    set(ax,'XAxisLocation','top','TickLength',[0 0],'YDir','reverse');
    set(ax,'XTick',1.5:off,'XTickLabels',data.FirmNames,'XTickLabelRotation',45,'YTick',1.5:off,'YTickLabels',data.FirmNames,'YTickLabelRotation',45);
    
    t = figure_title('Average Adjacency Matrix');
    t_position = get(t,'Position');
    set(t,'Position',[t_position(1) -0.0157 t_position(3)]);

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_centralities(data,id)

    seq = 1:data.N;
    
    [bc,order] = sort(data.BetweennessCentralitiesAverage);
    bc_names = data.FirmNames(order);
    [cc,order] = sort(data.ClosenessCentralitiesAverage);
    cc_names = data.FirmNames(order);
    [dc,order] = sort(data.DegreeCentralitiesAverage);
    dc_names = data.FirmNames(order);
    [ec,order] = sort(data.EigenvectorCentralitiesAverage);
    ec_names = data.FirmNames(order);
    [kc,order] = sort(data.KatzCentralitiesAverage);
    kc_names = data.FirmNames(order);
    [clc,order] = sort(data.ClusteringCoefficientsAverage);
    clc_names = data.FirmNames(order);

    f = figure('Name','Connectedness Measures > Average Centrality Measures','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    sub_1 = subplot(2,3,1);
    bar(sub_1,seq,bc,'FaceColor',[0.749 0.862 0.933]);
    set(sub_1,'XTickLabel',bc_names);
    title('Betweenness Centrality');
    
    sub_2 = subplot(2,3,2);
    bar(sub_2,seq,cc,'FaceColor',[0.749 0.862 0.933]);
    set(sub_2,'XTickLabel',cc_names);
    title('Closeness Centrality');
    
    sub_3 = subplot(2,3,3);
    bar(sub_3,seq,dc,'FaceColor',[0.749 0.862 0.933]);
    set(sub_3,'XTickLabel',dc_names);
    title('Degree Centrality');
    
    sub_4 = subplot(2,3,4);
    bar(sub_4,seq,ec,'FaceColor',[0.749 0.862 0.933]);
    set(sub_4,'XTickLabel',ec_names);
    title('Eigenvector Centrality');
    
    sub_5 = subplot(2,3,5);
    bar(sub_5,seq,kc,'FaceColor',[0.749 0.862 0.933]);
    set(sub_5,'XTickLabel',kc_names);
    title('Katz Centrality');

    sub_6 = subplot(2,3,6);
    bar(sub_6,seq,clc,'FaceColor',[0.749 0.862 0.933]);
    set(sub_6,'XTickLabel',clc_names);
    title('Clustering Coefficient');
    
    set([sub_1 sub_2 sub_3 sub_4 sub_5 sub_6],'XLim',[0 (data.N + 1)],'XTick',seq,'XTickLabelRotation',90);

    t = figure_title('Average Centrality Measures');
    t_position = get(t,'Position');
    set(t,'Position',[t_position(1) -0.0157 t_position(3)]);
    
    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end
