% [INPUT]
% data = A structure representing the dataset.
% out_temp = A string representing the full path to the Excel spreadsheet used as a template for the results file.
% out_file = A string representing the full path to the Excel spreadsheet to which the results are written, eventually replacing the previous ones.
% bandwidth = An integer greater than or equal to 30 representing the bandwidth (dimension) of each rolling window (optional, default=252).
% f = A float (0.2:0.1:0.8) representing the percentage of components to include in the computation of the absorption ratio (optional, default=0.20).
% analyze = A boolean that indicates whether to analyse the results and display plots (optional, default=false).
%
% [OUTPUT]
% result = A structure representing the original dataset inclusive of intermediate and final calculations.

function result = run_component(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data',@(x)validateattributes(x,{'struct'},{'nonempty'}));
        ip.addRequired('out_temp',@(x)validateattributes(x,{'char'},{'nonempty','size',[1,NaN]}));
        ip.addRequired('out_file',@(x)validateattributes(x,{'char'},{'nonempty','size',[1,NaN]}));
        ip.addOptional('bandwidth',252,@(x)validateattributes(x,{'numeric'},{'scalar','integer','real','finite','>=',30}));
        ip.addOptional('f',0.2,@(x)validateattributes(x,{'double','single'},{'scalar','real','finite','>=',0.2,'<=',0.8}));
        ip.addOptional('analyze',false,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    data = validate_data(ipr.data);
    out_temp = validate_template(ipr.out_temp);
    out_file = validate_output(ipr.out_file);
    f = validate_f(ipr.f);
    
    result = run_component_internal(data,out_temp,out_file,ipr.bandwidth,f,ipr.analyze);

end

function result = run_component_internal(data,out_temp,out_file,bandwidth,f,analyze)

    result = [];
    
    bar = waitbar(0,'Calculating component measures...','CreateCancelBtn',@(src,event)setappdata(gcbf(),'Stop',true));
    setappdata(bar,'Stop',false);
    
    windows = extract_rolling_windows(data.FirmReturns,bandwidth);
    windows_len = length(windows);

    data = data_initialize(data,windows_len,bandwidth,f);
    
	rng_settings = rng();
    rng(0);
    
    stopped = false;
    e = [];

    futures(1:windows_len) = parallel.FevalFuture;
    futures_max = 0;
    futures_results = cell(windows_len,1);
    
    for i = 1:windows_len
        futures(i) = parfeval(@main_loop,1,windows{i},data.Components);
    end
    
    try
        for i = 1:windows_len
            if (getappdata(bar,'Stop'))
            	stopped = true;
                break;
            end
            
            [future_index,value] = fetchNext(futures);
            futures_results{future_index} = value;
            
            futures_max = max([future_index futures_max]);
            waitbar((futures_max - 1) / windows_len,bar);

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
    
    rng(rng_settings);
    
    if (~isempty(e))
        delete(bar);
        rethrow(e);
    end
    
    if (stopped)
        delete(bar);
        return;
    end
    
    data = data_finalize(data,futures_results);

    waitbar(100,bar,'Writing component measures...');
    write_results(out_temp,out_file,data);
	delete(bar);

    if (analyze)        
        plot_indicators(data);
        plot_pca(data);
    end
    
    result = data;

end

%% DATA

function data = data_initialize(data,windows_len,bandwidth,f)

    data.Bandwidth = bandwidth;
    data.Components = round(data.N * f);
    data.F = f;
    data.Windows = windows_len;

    data.AbsorptionRatio = NaN(data.T,1);
    data.CorrelationSurprise = NaN(data.T,1);
    data.TurbulenceIndex = NaN(data.T,1);

    data.PCACoefficients = cell(windows_len,1);
    data.PCAExplained = cell(windows_len,1);
    data.PCAExplainedSums = NaN(data.T,4);
    data.PCAScores = cell(windows_len,1);

end

function data = data_finalize(data,futures_results)

    for i = 1:data.Windows
        futures_result = futures_results{i};
        futures_result_off = data.Bandwidth + i - 1;

        data.AbsorptionRatio(futures_result_off) = futures_result.AbsorptionRatio;
        data.CorrelationSurprise(futures_result_off) = futures_result.CorrelationSurprise;
        data.TurbulenceIndex(futures_result_off) = futures_result.TurbulenceIndex;
        
        data.PCACoefficients{i} = futures_result.PCACoefficients;
        data.PCAExplained{i} = futures_result.PCAExplained;
        data.PCAExplainedSums(futures_result_off,:) = fliplr([cumsum([futures_result.PCAExplained(1) futures_result.PCAExplained(2) futures_result.PCAExplained(3)]) 100]);
        data.PCAScores{i} = futures_result.PCAScores;
    end

    [coefficients,scores,explained] = calculate_pca(data.FirmReturns);
    data.PCACoefficientsOverall = coefficients;
    data.PCAExplainedOverall = explained;
    data.PCAExplainedSumsOverall = fliplr([cumsum([explained(1) explained(2) explained(3)]) 100]);
    data.PCAScoresOverall = scores;

end

function f = validate_f(f)

    f_seq = 0.2:0.1:0.8;

    if (~ismember(f,f_seq))
        error(['The f parameter must have one of the following values: ' strjoin(arrayfun(@(x)sprintf('%.1f',x),0.2:0.1:0.8,'UniformOutput',false),', ') '.']);
    end

end

function data = validate_data(data)

    fields = {'Full', 'T', 'N', 'DatesNum', 'DatesStr', 'MonthlyTicks', 'IndexName', 'IndexReturns', 'FirmNames', 'FirmReturns', 'Capitalizations', 'CapitalizationsLagged', 'Liabilities', 'SeparateAccounts', 'StateVariables', 'Groups', 'GroupDelimiters', 'GroupNames'};
    
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
    
    sheets = {'Indicators' 'PCA Overall Explained' 'PCA Overall Coefficients' 'PCA Overall Scores'};

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

    firm_names = data.FirmNames';

    vars = [data.DatesStr num2cell(data.AbsorptionRatio) num2cell(data.CorrelationSurprise) num2cell(data.TurbulenceIndex)];
    labels = {'Date' 'Absorption_Ratio' 'Correlation_Surprise' 'Turbulence_Index'};
    t1 = cell2table(vars,'VariableNames',labels);
    writetable(t1,out_file,'FileType','spreadsheet','Sheet','Indicators','WriteRowNames',true);

    vars = [num2cell(1:data.N)' num2cell(data.PCAExplainedOverall)];
    labels = {'PC' 'ExplainedVariance'};
    t2 = cell2table(vars,'VariableNames',labels);
    writetable(t2,out_file,'FileType','spreadsheet','Sheet','PCA Overall Explained','WriteRowNames',true);

    vars = [firm_names num2cell(data.PCACoefficientsOverall)];
    labels = {'Firms' data.FirmNames{:,:}};
    t3 = cell2table(vars,'VariableNames',labels);
    writetable(t3,out_file,'FileType','spreadsheet','Sheet','PCA Overall Coefficients','WriteRowNames',true);
    
    vars = [data.DatesStr num2cell(data.PCAScoresOverall)];
    labels = {'Date' data.FirmNames{:,:}};
    t4 = cell2table(vars,'VariableNames',labels);
    writetable(t4,out_file,'FileType','spreadsheet','Sheet','PCA Overall Scores','WriteRowNames',true);
    
end

%% MEASURES

function window_results = main_loop(window,components)

    window_results = struct();

    [ar,cs,ti] = calculate_indicators(window,components);
    window_results.AbsorptionRatio = ar;
    window_results.CorrelationSurprise = cs;
    window_results.TurbulenceIndex = ti;
    
    [coefficients,scores,explained] = calculate_pca(window);
    window_results.PCACoefficients = coefficients;
    window_results.PCAExplained = explained;
    window_results.PCAScores = scores;

end

function [ar,cs,ti] = calculate_indicators(data,components)

    zeros_indices = find(~data);
    data(zeros_indices) = (((1e-10 - 1e-8) .* rand(numel(zeros_indices),1)) + 1e-8);

    c = cov(data);
    bm = eye(size(c)) .* diag(c);
    e = eigs(c,size(c,1));

    v = data(end,:) - mean(data(1:end-1,:),1);
    vt = v.';

    ar = sum(e(1:components)) / trace(c);
    cs = ((v / c) * vt) / ((v / bm) * vt);
    ti = (v / c) * vt;

end

function [coefficients,scores,explained] = calculate_pca(data)

    data_normalized = data;

    for i = 1:size(data_normalized,2)
        c = data_normalized(:,i);

        m = mean(c);

        s = std(c);
        s(s == 0) = 1;

        data_normalized(:,i) = (c - m) / s;
    end

    data_normalized(isnan(data_normalized)) = 0;

    [coefficients,scores,~,~,explained] = pca(data_normalized,'Economy',false);

end

%% PLOTTING

function plot_indicators(data)

    alpha = 2 / (data.Bandwidth + 1);
    nans = NaN(data.Bandwidth-1,1);

    ti = data.TurbulenceIndex(data.Bandwidth:end);
    ti_ma = filter(alpha,[1 (alpha - 1)],ti(2:end,:),(1 - alpha) * ti(1,:));
	ti_ma = [nans; ti(1,:); ti_ma];
    
    cs = data.CorrelationSurprise(data.Bandwidth:end);
    cs_ma = filter(alpha,[1 (alpha - 1)],cs(2:end,:),(1 - alpha) * cs(1,:));
	cs_ma = [nans; cs(1,:); cs_ma];

    f = figure('Name','Component Indicators','Units','normalized','Position',[100 100 0.85 0.85]);
    
    colors = get(gca(),'ColorOrder');
    color = colors(1,:);

    sub_1 = subplot(2,2,[1 3]);
    plot(sub_1,data.DatesNum,data.AbsorptionRatio,'Color',color);
    t1 = title(sub_1,['Absorption Ratio (f=' sprintf('%.1f',data.F) ')']);
    set(t1,'Units','normalized');
    t1_position = get(t1,'Position');
    set(t1,'Position',[0.4783 t1_position(2) t1_position(3)]);

    sub_2 = subplot(2,2,2);
    p2 = plot(sub_2,data.DatesNum,data.TurbulenceIndex,'Color',[0.65 0.65 0.65]);
    p2.Color(4) = 0.35;
    hold on;
        plot(sub_2,data.DatesNum,ti_ma,'Color',color);
    hold off;
    t2 = title(sub_2,'Turbulence Index (EWMA)');
    set(t2,'Units','normalized');
    t2_position = get(t2,'Position');
    set(t2,'Position',[0.4783 t2_position(2) t2_position(3)]);
    
    sub_3 = subplot(2,2,4);
    p3 = plot(sub_3,data.DatesNum,data.CorrelationSurprise,'Color',[0.65 0.65 0.65]);
    p3.Color(4) = 0.35;
    hold on;
        plot(sub_3,data.DatesNum,cs_ma,'Color',color);
    hold off;
    t3 = title(sub_3,'Correlation Surprise (EWMA)');
    set(t3,'Units','normalized');
    t3_position = get(t3,'Position');
    set(t3,'Position',[0.4783 t3_position(2) t3_position(3)]);
    
    set([sub_1 sub_2 sub_3],'XLim',[data.DatesNum(data.Bandwidth) data.DatesNum(end)],'XTickLabelRotation',45);

    if (data.MonthlyTicks)
        datetick(sub_1,'x','mm/yyyy','KeepLimits','KeepTicks');
        datetick(sub_2,'x','mm/yyyy','KeepLimits','KeepTicks');
        datetick(sub_3,'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        datetick(sub_1,'x','yyyy','KeepLimits');
        datetick(sub_2,'x','yyyy','KeepLimits');
        datetick(sub_3,'x','yyyy','KeepLimits');
    end

    t = figure_title('Component Indicators');
    t_position = get(t,'Position');
    set(t,'Position',[t_position(1) -0.0157 t_position(3)]);
    
    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_pca(data)

    coefficients = data.PCACoefficientsOverall(:,1:3);
    [coefficients_rows,coefficients_columns] = size(coefficients);
    [~,indices] = max(abs(coefficients),[],1);
    coefficients_max_len = sqrt(max(sum(coefficients.^2,2)));
    coefficients_columns_sign = sign(coefficients(indices + (0:coefficients_rows:((coefficients_columns-1)*coefficients_rows))));
    coefficients = bsxfun(@times,coefficients,coefficients_columns_sign);

    scores = data.PCAScoresOverall(:,1:3);
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
    y_labels = arrayfun(@(x)sprintf('%d%%',x),y_ticks,'UniformOutput',false);
    
    f = figure('Name','Principal Component Analysis','Units','normalized');

    sub_1 = subplot(1,2,1);
    line_1 = line(x_area(1:2,:),y_area(1:2,:),z_area(1:2,:),'LineStyle','-','Marker','none');
    line_2 = line(x_area(2:3,:),y_area(2:3,:),z_area(2:3,:),'LineStyle','none','Marker','.');
    set([line_1 line_2],'Color','b');
    line(x_points,y_points,z_points,'Color','r','LineStyle','none','Marker','.');
    view(sub_1,coefficients_columns);
    grid on;
    line([limits_low limits_high NaN 0 0 NaN 0 0],[0 0 NaN limits_low limits_high NaN 0 0],[0 0 NaN 0 0 NaN limits_low limits_high],'Color','k');
    axis tight;
    xlabel(sub_1,'PC 1');
    ylabel(sub_1,'PC 2');
    zlabel(sub_1,'PC 3');
    title('Coefficients & Scores');

    sub_2 = subplot(1,2,2);
    area_1 = area(sub_2,data.DatesNum,data.PCAExplainedSums(:,1),'FaceColor',[0.7 0.7 0.7]);
    hold on;
        area_2 = area(sub_2,data.DatesNum,data.PCAExplainedSums(:,2),'FaceColor','g');
        area_3 = area(sub_2,data.DatesNum,data.PCAExplainedSums(:,3),'FaceColor','b');
        area_4 = area(sub_2,data.DatesNum,data.PCAExplainedSums(:,4),'FaceColor','r');
    hold off;
    datetick('x','yyyy','KeepLimits');
    set([area_1 area_2 area_3 area_4],'EdgeColor','none');
    set(sub_2,'XLim',[data.DatesNum(data.Bandwidth) data.DatesNum(end)],'XTick',[],'YLim',[y_ticks(1) y_ticks(end)],'YTick',y_ticks,'YTickLabel',y_labels);
    legend(sub_2,sprintf('PC 4-%d',data.N),'PC 3','PC 2','PC 1','Location','southeast');
    title('Explained Variance');

    t = figure_title('Principal Component Analysis');
    t_position = get(t,'Position');
    set(t,'Position',[t_position(1) -0.0157 t_position(3)]);
    
    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end
