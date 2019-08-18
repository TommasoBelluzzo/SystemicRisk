% [INPUT]
% data         = A structure representing the dataset.
% out_temp     = A string representing the full path to the Excel spreadsheet used as a template for the results file.
% out_file     = A string representing the full path to the Excel spreadsheet to which the results are written, eventually replacing the previous ones.
% significance = A float representing the statistical significance threshold for the linear Granger-causality test (optional, default=0.05).
% robust       = A boolean indicating whether to use robust p-values (optional, default=true).
% bandwidth    = An integer representing the bandwidth (dimension) of each rolling window (optional, default=252).
% analyze      = A boolean that indicates whether to analyse the results and display plots (optional, default=false).

function run_network(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data',@(x)validateattributes(x,{'struct'},{'nonempty'}));
        ip.addRequired('out_temp',@(x)validateattributes(x,{'char'},{'nonempty','size',[1,NaN]}));
        ip.addRequired('out_file',@(x)validateattributes(x,{'char'},{'nonempty','size',[1,NaN]}));
        ip.addOptional('significance',0.05,@(x)validateattributes(x,{'double','single'},{'scalar','real','finite','>',0,'<=',0.20}));
        ip.addOptional('robust',true,@(x)validateattributes(x,{'logical'},{'scalar'}));
        ip.addOptional('bandwidth',252,@(x)validateattributes(x,{'numeric'},{'vector','integer','real','finite','>=',30}));
        ip.addOptional('analyze',false,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    data = validate_data(ipr.data);
    out_temp = validate_template(ipr.out_temp);
    out_file = validate_output(ipr.out_file);
    
    run_network_internal(data,out_temp,out_file,ipr.significance,ipr.robust,ipr.bandwidth,ipr.analyze);

end

function run_network_internal(data,out_temp,out_file,significance,robust,bandwidth,analyze)

    bar = waitbar(0,'Calculating network measures...','CreateCancelBtn','setappdata(gcbf,''Stop'',true)');
    setappdata(bar,'Stop',false);
    
    data = data_initialize(data,significance,robust,bandwidth);

    windows = get_rolling_windows(data.FrmsRet,bandwidth);
    windows_len = length(windows);
    windows_diff = data.Obs - windows_len;
    
    try
        for i = 1:windows_len
            waitbar(((i - 1) / windows_len),bar,sprintf('Calculating network measures for window %d of %d...',i,windows_len));
            
            if (getappdata(bar,'Stop'))
                delete(bar);
                return;
            end
            
            window = windows{i,1};
            window_offset = i + windows_diff;

            adjacency_matrix = calculate_adjacency_matrix(window,data.Significance,data.Robust);
            data.AdjacencyMatrices{window_offset} = adjacency_matrix;
            
            [dci,number_io,number_ioo] = calculate_connectedness(adjacency_matrix,data.GrpsDel);
            
            data.DCI(window_offset) = dci;
            data.NumberIO(window_offset) = number_io;
            data.NumberIOO(window_offset) = number_ioo;
            
            [bc,cc,dc,ec,kc,clustering_coefficients] = calculate_centralities(adjacency_matrix);
            
            data.BetCen(window_offset,:) = bc;
            data.CloCen(window_offset,:) = cc;
            data.DegCen(window_offset,:) = dc;
            data.EigCen(window_offset,:) = ec;
            data.KatCen(window_offset,:) = kc;
            data.CluCoe(window_offset,:) = clustering_coefficients;

            window_normalized = window;
            
            for j = 1:size(window_normalized,2)
                window_normalized_j = window_normalized(:,j);
                window_normalized_j = window_normalized_j - nanmean(window_normalized_j);
                window_normalized(:,j) = window_normalized_j / nanstd(window_normalized_j);
            end
            
            window_normalized(isnan(window_normalized)) = 0;

            [pca_coefficients,pca_scores,~,~,pca_explained] = pca(window_normalized,'Economy',false);

            data.PCACoe{window_offset} = pca_coefficients;
            data.PCAExp{window_offset} = pca_explained;
            data.PCASco{window_offset} = pca_scores;

            if (getappdata(bar,'Stop'))
                delete(bar);
                return;
            end
            
            waitbar((i / windows_len),bar,sprintf('Calculating network measures for window %d of %d...',i,windows_len));
        end

        data = data_finalize(data,windows_diff);

        waitbar(100,bar,'Writing network measures...');
        write_results(out_temp,out_file,data);
        
        delete(bar);
    catch e
        delete(bar);
        rethrow(e);
    end

    if (analyze)        
        plot_indicators(data);
        plot_network(data);
        plot_centralities(data);
        plot_pca(data);
    end

end

%%%%%%%%
% DATA %
%%%%%%%%

function data = data_initialize(data,significance,robust,bandwidth)

    data.Bandwidth = bandwidth;
    data.Robust = robust;
    data.Significance = significance;

    data.AdjacencyMatrices = cell(data.Obs,1);
    
    data.DCI = NaN(data.Obs,1);
    data.NumberIO = NaN(data.Obs,1);
    data.NumberIOO = NaN(data.Obs,1);

    data.BetCen = NaN(data.Obs,data.Frms);
    data.CloCen = NaN(data.Obs,data.Frms);
    data.CluCoe = NaN(data.Obs,data.Frms);
    data.DegCen = NaN(data.Obs,data.Frms);
    data.EigCen = NaN(data.Obs,data.Frms);
    data.KatCen = NaN(data.Obs,data.Frms);

    data.PCACoe = cell(data.Obs,1);
    data.PCAExp = cell(data.Obs,1);
    data.PCASco = cell(data.Obs,1);

end

function data = data_finalize(data,win_dif)

    data.WinOff = win_dif + 1;

    windows_sequence =  data.WinOff:data.Obs;
    windows_sequence_len = length(windows_sequence);

    data.AdjMatAvg = sum(cat(3,data.AdjacencyMatrices{windows_sequence}),3) ./ windows_sequence_len;
    
    thre = mean(mean(data.AdjMatAvg));
    data.AdjMatAvg(data.AdjMatAvg < thre) = 0;
    data.AdjMatAvg(data.AdjMatAvg >= thre) = 1;

    data.BetCenAvg = sum(data.BetCen(windows_sequence,:),1) ./ windows_sequence_len;
    data.CloCenAvg = sum(data.CloCen(windows_sequence,:),1) ./ windows_sequence_len;
    data.CluCoeAvg = sum(data.CluCoe(windows_sequence,:),1) ./ windows_sequence_len;
    data.DegCenAvg = sum(data.DegCen(windows_sequence,:),1) ./ windows_sequence_len;
    data.EigCenAvg = sum(data.EigCen(windows_sequence,:),1) ./ windows_sequence_len;
    data.KatCenAvg = sum(data.KatCen(windows_sequence,:),1) ./ windows_sequence_len;

    data.PCACoeAvg = sum(cat(3,data.PCACoe{windows_sequence}),3) ./ windows_sequence_len;
    data.PCAExpAvg = sum(cat(3,data.PCAExp{windows_sequence}),3) ./ windows_sequence_len;
    data.PCAScoAvg = sum(cat(3,data.PCASco{windows_sequence}),3) ./ windows_sequence_len;
    data.PCAExpSum = NaN(data.Obs,4);
    
    for i = windows_sequence
        exp = data.PCAExp{i};
        data.PCAExpSum(i,:) = fliplr([cumsum([exp(1) exp(2) exp(3)]) 100]);
    end

end

function windows = get_rolling_windows(data,bandwidth)

    t = size(data,1);
    
    if (bandwidth >= t)
        windows = cell(1,1);
        windows{1} = data;
        return;
    end

    limit = t - bandwidth + 1;
    windows = cell(limit,1);

    for i = 1:limit
        windows{i} = data(i:bandwidth+i-1,:);
    end

end

function data = validate_data(data)

    fields = {'DatesNum', 'DatesStr', 'Frms', 'FrmsCap', 'FrmsCapLag', 'FrmsLia', 'FrmsSep', 'FrmsNam', 'FrmsRet', 'Full', 'Grps', 'GrpsDel', 'GrpsNam', 'IdxNam', 'IdxRet', 'Obs', 'StateVariables'};

    for i = 1:numel(fields)
        if (~isfield(data,fields{i}))
            error('The dataset does not contain all the required data.');
        end
    end
    
end

function out_file = validate_output(out_file)

    [path,name,extension] = fileparts(out_file);

    if (~strcmp(extension,'.xlsx'))
        out_file = fullfile(path,[name extension '.xlsx']);
    end
    
end

function out_temp = validate_template(out_temp)

    if (exist(out_temp,'file') == 0)
        error('The template file could not be found.');
    end
    
    if (ispc())
        [file_status,file_sheets,file_format] = xlsfinfo(out_temp);
        
        if (isempty(file_status) || ~strcmp(file_format,'xlOpenXMLWorkbook'))
            error('The dataset file is not a valid Excel spreadsheet.');
        end
    else
        [file_status,file_sheets] = xlsfinfo(out_temp);
        
        if (isempty(file_status))
            error('The dataset file is not a valid Excel spreadsheet.');
        end
    end
    
    sheets = {'Indicators' 'Average Adjacency Matrix' 'Average Centrality Measures' 'PCA Explained Variances' 'PCA Average Coefficients' 'PCA Average Scores'};

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

function write_results(out_temp,out_file,data)

    [out_file_path,~,~] = fileparts(out_file);

    if (exist(out_file_path,'dir') ~= 7)
        mkdir(out_file_path);
    end

    if (exist(out_file,'file') == 2)
        delete(out_file);
    end
    
    copy_result = copyfile(out_temp,out_file,'f');
    
    if (copy_result == 0)
        error('The results file could not be created from the template file.');
    end

    firm_names = data.FrmsNam';
    
    vars = [data.DatesStr num2cell(data.DCI) num2cell(data.NumberIO) num2cell(data.NumberIOO)];
    labels = {'Date' 'DCI' 'NumIO' 'NumIOO'};
    t1 = cell2table(vars,'VariableNames',labels);
    writetable(t1,out_file,'FileType','spreadsheet','Sheet','Indicators','WriteRowNames',true);

    vars = [firm_names num2cell(data.AdjMatAvg)];
    labels = {'Firms' data.FrmsNam{:,:}};
    t2 = cell2table(vars,'VariableNames',labels);
    writetable(t2,out_file,'FileType','spreadsheet','Sheet','Average Adjacency Matrix','WriteRowNames',true);

    vars = [firm_names num2cell(data.BetCenAvg') num2cell(data.CloCenAvg') num2cell(data.DegCenAvg') num2cell(data.EigCenAvg') num2cell(data.KatCenAvg') num2cell(data.CluCoeAvg')];
    labels = {'Firms' 'BetweennessCentrality' 'ClosenessCentrality' 'DegreeCentrality' 'EigenvectorCentrality' 'KatzCentrality' 'ClusteringCoefficient'};
    t3 = cell2table(vars,'VariableNames',labels);
    writetable(t3,out_file,'FileType','spreadsheet','Sheet','Average Centrality Measures','WriteRowNames',true);

    vars = [num2cell(1:data.Frms)' num2cell(data.PCAExpAvg)];
    labels = {'PC' 'ExplainedVariance'};
    t4 = cell2table(vars,'VariableNames',labels);
    writetable(t4,out_file,'FileType','spreadsheet','Sheet','PCA Explained Variances','WriteRowNames',true);

    vars = [firm_names num2cell(data.PCACoeAvg)];
    labels = {'Firms' data.FrmsNam{:,:}};
    t5 = cell2table(vars,'VariableNames',labels);
    writetable(t5,out_file,'FileType','spreadsheet','Sheet','PCA Average Coefficients','WriteRowNames',true);
    
    vars = num2cell(data.PCAScoAvg);
    labels = data.FrmsNam;
    t6 = cell2table(vars,'VariableNames',labels);
    writetable(t6,out_file,'FileType','spreadsheet','Sheet','PCA Average Scores','WriteRowNames',true);

end

%%%%%%%%%%%%
% MEASURES %
%%%%%%%%%%%%

function [bc,cc,dc,ec,kc,clustering_coefficients] = calculate_centralities(adjacency_matrix)

    adjacency_matrix_len = length(adjacency_matrix);

    bc = calculate_betweenness_centrality(adjacency_matrix,adjacency_matrix_len);
    [degrees,dc] = calculate_degree_centrality(adjacency_matrix,adjacency_matrix_len);
    cc = calculate_closeness_centrality(adjacency_matrix,adjacency_matrix_len);
    ec = calculate_eigenvector_centrality(adjacency_matrix);
    kc = calculate_katz_centrality(adjacency_matrix,adjacency_matrix_len);
    clustering_coefficients = calculate_clustering_coefficient(adjacency_matrix,adjacency_matrix_len,degrees);

end

function [dci,number_io,number_ioo] = calculate_connectedness(adjacency_matrix,group_delimiters)

    n = length(adjacency_matrix);

    links_current = sum(sum(adjacency_matrix));
    links_max = (n ^ 2) - n;
    dci = links_current / links_max;

    number_i = zeros(n,1);
    number_o = zeros(n,1);
    
    for i = 1:n     
        number_i(i) = sum(adjacency_matrix(:,i));
        number_o(i) = sum(adjacency_matrix(i,:));
    end

    number_io = sum(number_i) + sum(number_o);
    
    if (isempty(group_delimiters))
        number_ioo = NaN;
    else
        groups_len = length(group_delimiters);
        number_ifo = zeros(n,1);
        number_oto = zeros(n,1);
        
        for i = 1:n
            group_1 = group_delimiters(1);
            group_n = group_delimiters(groups_len);
            
            if (i <= group_1)
                group_begin = 1;
                group_end = group_1;
            elseif (i > group_n)
                group_begin = group_n + 1;
                group_end = n;
            else
                for j = 1:groups_len-1
                    group_j0 = group_delimiters(j);
                    group_j1 = group_delimiters(j+1);

                    if ((i > group_j0) && (i <= group_j1))
                        group_begin = group_j0 + 1;
                        group_end = group_j1;
                    end
                end
            end

            number_ifo(i) = number_i(i) - sum(adjacency_matrix(group_begin:group_end,i));
            number_oto(i) = number_o(i) - sum(adjacency_matrix(i,group_begin:group_end));
        end

        number_ioo = sum(number_ifo) + sum(number_oto);
    end

end

function bc = calculate_betweenness_centrality(adjacency_matrix,adjacency_matrix_len)

    bc = zeros(1,adjacency_matrix_len);

    for i = 1:adjacency_matrix_len
        depth = 0;
        nsp = accumarray([1 i],1,[1 adjacency_matrix_len]);
        bfs = false(250,adjacency_matrix_len);
        fringe = adjacency_matrix(i,:);

        while ((nnz(fringe) > 0) && (depth <= 250))
            depth = depth + 1;
            nsp = nsp + fringe;
            bfs(depth,:) = logical(fringe);
            fringe = (fringe * adjacency_matrix) .* ~nsp;
        end

        [rows,cols,v] = find(nsp);
        v = 1 ./ v;
        
        nsp_inv = accumarray([rows.' cols.'],v,[1 adjacency_matrix_len]);

        bcu = ones(1,adjacency_matrix_len);

        for depth = depth:-1:2
            w = (bfs(depth,:) .* nsp_inv) .* bcu;
            bcu = bcu + ((adjacency_matrix * w.').' .* bfs(depth-1,:)) .* nsp;
        end

        bc = bc + sum(bcu,1);
    end

    bc = bc - adjacency_matrix_len;
    bc = (bc .* 2) ./ ((adjacency_matrix_len - 1) * (adjacency_matrix_len - 2));

end

function cc = calculate_closeness_centrality(adjacency_matrix,adjacency_matrix_len)

    cc = zeros(1,adjacency_matrix_len);

    for i = 1:adjacency_matrix_len
        paths = dijkstra_shortest_paths(adjacency_matrix,adjacency_matrix_len,i);
        paths_sum = sum(paths(~isinf(paths)));
        
        if (paths_sum ~= 0)
            cc(i) = 1 / paths_sum;
        end
    end

    cc = cc .* (adjacency_matrix_len - 1);

end

function clustering_coefficients = calculate_clustering_coefficient(adjacency_matrix,adjacency_matrix_len,degrees)

    if (issymmetric(adjacency_matrix))
        z = 2;
    else
        z = 1;
    end
    
    clustering_coefficients = zeros(adjacency_matrix_len,1);

    for i = 1:adjacency_matrix_len
        degi = degrees(i);

        if ((degi == 0) || (degi == 1))
            continue;
        end

        knei = find(adjacency_matrix(i,:) ~= 0);
        ksub = adjacency_matrix(knei,knei);

        if (issymmetric(ksub))
            ksub_sl = trace(ksub);
            
            if (ksub_sl == 0)
                edges = sum(sum(ksub)) / 2; 
            else
                edges = ((sum(sum(ksub)) - ksub_sl) / 2) + ksub_sl;
            end
        else
            edges = sum(sum(ksub));
        end

        clustering_coefficients(i) = ((z * edges) / degi) / (degi - 1);

    end

end

function [degrees,degree_centrality] = calculate_degree_centrality(adjacency_matrix,adjacency_matrix_len)

    if (issymmetric(adjacency_matrix))
        degrees = sum(adjacency_matrix) + sum(adjacency_matrix.');
    else
        degrees = sum(adjacency_matrix) + diag(adjacency_matrix).';
    end
    
    degree_centrality = degrees ./ (adjacency_matrix_len - 1);

end

function ec = calculate_eigenvector_centrality(adjacency_matrix)

	[eigen_vector,eigen_values] = eig(adjacency_matrix);
    [~,indices] = max(diag(eigen_values));

    ec = abs(eigen_vector(:,indices)).';
    ec = ec ./ sum(ec);

end

function kc = calculate_katz_centrality(adjacency_matrix,adjacency_matrix_len)

    kc = (eye(adjacency_matrix_len) - (adjacency_matrix .* 0.1)) \ ones(adjacency_matrix_len,1);
    kc = kc.' ./ (sign(sum(kc)) * norm(kc,'fro'));

end

function paths = dijkstra_shortest_paths(adjm,adjm_len,node)

    paths = Inf(1,adjm_len);
    paths(node) = 0;

    adjm_seq = 1:adjm_len;

    while (~isempty(adjm_seq))
        [~,idx] = min(paths(adjm_seq));
        adjm_seq_idx = adjm_seq(idx);

        for i = 1:length(adjm_seq)
            adjm_seq_i = adjm_seq(i);

            adjm_off = adjm(adjm_seq_idx,adjm_seq_i);
            sum_off = adjm_off + paths(adjm_seq_idx);
            
            if ((adjm_off > 0) && (paths(adjm_seq_i) > sum_off))
                paths(adjm_seq_i) = sum_off;
            end
        end

        adjm_seq = setdiff(adjm_seq,adjm_seq_idx);
    end

end

%%%%%%%%%%%%
% PLOTTING %
%%%%%%%%%%%%

function plot_indicators(data)

    f = figure('Name','Measures of Connectedness','Units','normalized','Position',[100 100 0.85 0.85]);
    
    sub_1 = subplot(2,1,1);
    plot(sub_1,data.DatesNum,data.DCI);
    t1 = title(sub_1,'Dynamic Causality Index');
    set(t1,'Units','normalized');
    t1_position = get(t1,'Position');
    set(t1,'Position',[0.4783 t1_position(2) t1_position(3)]);

    sub_2 = subplot(2,1,2);
    handle_area_1 = area(sub_2,data.DatesNum,data.NumberIO,'EdgeColor','none','FaceColor','b');
    hold on;
        if (data.Grps == 0)
            handle_area_2 = area(sub_2,data.DatesNum,data.NumberIO,'EdgeColor','none','FaceColor',[0.678 0.922 1]);
            area(sub_2,data.DatesNum,data.NumberIO,'EdgeColor','none','FaceColor','b');
        else
            handle_area_2 = area(sub_2,data.DatesNum,data.NumberIOO,'EdgeColor','none','FaceColor',[0.678 0.922 1]);
        end
    hold off;
    legend(sub_2,[handle_area_1 handle_area_2],'Num IO','Num IOO','Location','best');
    t2 = title(sub_2,'In & Out Connections');
    set(t2,'Units','normalized');
    t2_position = get(t2,'Position');
    set(t2,'Position',[0.4783 t2_position(2) t2_position(3)]);

    set([sub_1 sub_2],'XLim',[data.DatesNum(data.WinOff) data.DatesNum(end)],'XTickLabelRotation',45);
    
    indices_clean = ~isnan(data.NumberIO);

    if (length(unique(year(data.DatesNum(indices_clean)))) <= 3)
        datetick(sub_1,'x','mm/yyyy','KeepLimits','KeepTicks');
        datetick(sub_2,'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        datetick(sub_1,'x','yyyy','KeepLimits');
        datetick(sub_2,'x','yyyy','KeepLimits');
    end
    
    t = figure_title('Measures of Connectedness');
    t_position = get(t,'Position');
    set(t,'Position',[t_position(1) -0.0157 t_position(3)]);
    
    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_network(data)

    if (data.Grps == 0)
        grps_col = repmat(lines(1),data.Frms,1);
    else
        grps_col = zeros(data.Frms,3);
        grps_del_len = length(data.GrpsDel);
        grps_del_plu = data.GrpsDel + 1;
        grps_lin = lines(data.Grps);

        for i = 1:grps_del_len
            del = data.GrpsDel(i);

            if (i == 1)
                grps_col(1:del,:) = repmat(grps_lin(i,:),del,1);
            else
                del_prev = data.GrpsDel(i-1) + 1;
                grps_col(del_prev:del,:) = repmat(grps_lin(i,:),del-del_prev+1,1);
            end

            if (i == grps_del_len)
                grps_col(del+1:end,:) = repmat(grps_lin(i+1,:),data.Frms-del,1);
            end
        end
    end
    
    if (isempty(data.FrmsCap))
        wei = ones(1,data.Frms);
    else
        wei = mean(data.FrmsCap,1);
        wei = wei ./ mean(wei);
    end
    
    theta = linspace(0,(2 * pi),(data.Frms + 1)).';
    theta(end) = [];
    xy = [cos(theta) sin(theta)];
    [i,j] = find(data.AdjMatAvg);
    [~,idxs] = sort(max(i,j));
    i = i(idxs);
    j = j(idxs);
    x = [xy(i,1) xy(j,1)].';
    y = [xy(i,2) xy(j,2)].';

    fig = figure('Name','Network Graph','Units','normalized','Position',[100 100 0.85 0.85]);

    sub = subplot(100,1,10:100);
    
    hold on;
        for i = 1:size(x,2)
            idx = ismember(xy,[x(1,i) y(1,i)],'rows');
            plot(sub,x(:,i),y(:,i),'Color',grps_col(idx,:));
        end

        if (data.Grps == 0)
            for i = 1:size(xy,1)
                line(xy(i,1),xy(i,2),'Color',grps_col(i,:),'LineStyle','none','Marker','.','MarkerSize',(35 + (15 * wei(i))));
            end
        else
            lins = NaN(data.Grps,1);
            lins_off = 1;
            
            for i = 1:size(xy,1)
                grps_col_i = grps_col(i,:);
                line(xy(i,1),xy(i,2),'Color',grps_col_i,'LineStyle','none','Marker','.','MarkerSize',(35 + (15 * wei(i))));

                if ((i == 1) || any(grps_del_plu == i))
                    lins(lins_off) = line(xy(i,1),xy(i,2),'Color',grps_col_i,'LineStyle','none','Marker','.','MarkerSize',35);
                    lins_off = lins_off + 1;
                end
            end

            legend(sub,lins,data.GrpsNam,'Units','normalized','Position',[0.85 0.12 0.001 0.001]);
        end

        axis(sub,[-1 1 -1 1]);
        axis equal off;
    hold off;

    txts = text((xy(:,1) .* 1.05), (xy(:,2) .* 1.05),data.FrmsNam,'FontSize',10);
    set(txts,{'Rotation'},num2cell(theta * (180 / pi)));

    t = figure_title('Network Graph');
    t_pos = get(t,'Position');
    set(t,'Position',[t_pos(1) -0.0157 t_pos(3)]);
    
    pause(0.01);
    jfr = get(fig,'JavaFrame');
    set(jfr,'Maximized',true);

end

function plot_centralities(data)

    seq = 1:data.Frms;
    
    [betc_sor,order] = sort(data.BetCenAvg);
    betc_nam = data.FrmsNam(order);
    [cloc_sor,order] = sort(data.CloCenAvg);
    cloc_nam = data.FrmsNam(order);
    [degc_sor,order] = sort(data.DegCenAvg);
    degc_nam = data.FrmsNam(order);
    [eigc_sor,order] = sort(data.EigCenAvg);
    eigc_nam = data.FrmsNam(order);
    [katc_sor,order] = sort(data.KatCenAvg);
    katc_nam = data.FrmsNam(order);
    [cluc_sor,order] = sort(data.CluCoeAvg);
    cluc_nam = data.FrmsNam(order);

    f = figure('Name','Average Centrality Measures','Units','normalized','Position',[100 100 0.85 0.85]);

    sub_1 = subplot(2,3,1);
    bar(sub_1,seq,betc_sor,'FaceColor',[0.678 0.922 1]);
    set(sub_1,'XTickLabel',betc_nam);
    title('Betweenness Centrality');
    
    sub_2 = subplot(2,3,2);
    bar(sub_2,seq,cloc_sor,'FaceColor',[0.678 0.922 1]);
    set(sub_2,'XTickLabel',cloc_nam);
    title('Closeness Centrality');
    
    sub_3 = subplot(2,3,3);
    bar(sub_3,seq,degc_sor,'FaceColor',[0.678 0.922 1]);
    set(sub_3,'XTickLabel',degc_nam);
    title('Degree Centrality');
    
    sub_4 = subplot(2,3,4);
    bar(sub_4,seq,eigc_sor,'FaceColor',[0.678 0.922 1]);
    set(sub_4,'XTickLabel',eigc_nam);
    title('Eigenvector Centrality');
    
    sub_5 = subplot(2,3,5);
    bar(sub_5,seq,katc_sor,'FaceColor',[0.678 0.922 1]);
    set(sub_5,'XTickLabel',katc_nam);
    title('Katz Centrality');

    sub_6 = subplot(2,3,6);
    bar(sub_6,seq,cluc_sor,'FaceColor',[0.678 0.922 1]);
    set(sub_6,'XTickLabel',cluc_nam);
    title('Clustering Coefficient');
    
    set([sub_1 sub_2 sub_3 sub_4 sub_5 sub_6],'XLim',[0 (data.Frms + 1)],'XTick',seq,'XTickLabelRotation',90);

    t = figure_title('Average Centrality Measures');
    t_position = get(t,'Position');
    set(t,'Position',[t_position(1) -0.0157 t_position(3)]);
    
    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_pca(data)

    coe = data.PCACoeAvg(:,1:3);
    [coe_rows,coe_cols] = size(coe);

    sco = data.PCAScoAvg(:,1:3);
    sco_rows = size(sco,1);
    
    [~,idx] = max(abs(coe),[],1);
    coe_max_len = sqrt(max(sum(coe.^2,2)));
    cols_sig = sign(coe(idx + (0:coe_rows:((coe_cols-1)*coe_rows))));

    coe = bsxfun(@times,coe,cols_sig);
    sco = bsxfun(@times,(coe_max_len .* (sco ./ max(abs(sco(:))))),cols_sig);
    
    ar_beg = zeros(coe_rows,1);
    ar_end = NaN(coe_rows,1);
    x_ar = [ar_beg coe(:,1) ar_end]';
    y_ar = [ar_beg coe(:,2) ar_end]';
    z_ar = [ar_beg coe(:,3) ar_end]';
    
    ar_end = NaN(sco_rows,1);
    x_pts = [sco(:,1) ar_end]';
    y_pts = [sco(:,2) ar_end]';
    z_pts = [sco(:,3) ar_end]';

    lim_hi = 1.1 * max(abs(coe(:)));
    lim_lo = -lim_hi;
    
    y_tcks = 0:10:100;
    y_lbls = arrayfun(@(x)sprintf('%d%%',x),y_tcks,'UniformOutput',false);
    
    fig = figure('Name','Principal Component Analysis','Units','normalized');

    sub_1 = subplot(1,2,1);
    lin_1 = line(x_ar(1:2,:),y_ar(1:2,:),z_ar(1:2,:),'LineStyle','-','Marker','none');
    lin_2 = line(x_ar(2:3,:),y_ar(2:3,:),z_ar(2:3,:),'LineStyle','none','Marker','.');
    set([lin_1 lin_2],'Color','b');
    line(x_pts,y_pts,z_pts,'Color','r','LineStyle','none','Marker','.');
    view(sub_1,coe_cols);
    grid on;
    line([lim_lo lim_hi NaN 0 0 NaN 0 0],[0 0 NaN lim_lo lim_hi NaN 0 0],[0 0 NaN 0 0 NaN lim_lo lim_hi],'Color','k');
    axis tight;
    xlabel(sub_1,'PC 1');
    ylabel(sub_1,'PC 2');
    zlabel(sub_1,'PC 3');
    title('Coefficients & Scores');

    sub_2 = subplot(1,2,2);
    ar_1 = area(sub_2,data.DatesNum,data.PCAExpSum(:,1),'FaceColor',[0.7 0.7 0.7]);
    hold on;
        ar_2 = area(sub_2,data.DatesNum,data.PCAExpSum(:,2),'FaceColor','g');
        ar_3 = area(sub_2,data.DatesNum,data.PCAExpSum(:,3),'FaceColor','b');
        ar_4 = area(sub_2,data.DatesNum,data.PCAExpSum(:,4),'FaceColor','r');
    hold off;
    datetick('x','yyyy','KeepLimits');
    set([ar_1 ar_2 ar_3 ar_4],'EdgeColor','none');
    set(sub_2,'XLim',[data.DatesNum(data.WinOff) data.DatesNum(end)],'YLim',[y_tcks(1) y_tcks(end)],'YTick',y_tcks,'YTickLabel',y_lbls);
    legend(sub_2,sprintf('PC 4-%d',data.Frms),'PC 3','PC 2','PC 1','Location','southeast');
    title('Explained Variance');

    t = figure_title('Principal Component Analysis');
    t_pos = get(t,'Position');
    set(t,'Position',[t_pos(1) -0.0157 t_pos(3)]);
    
    pause(0.01);

    jfr = get(fig,'JavaFrame');
    set(jfr,'Maximized',true);

end
