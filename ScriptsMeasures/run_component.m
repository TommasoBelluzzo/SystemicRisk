% [INPUT]
% data = A structure representing the dataset.
% temp = A string representing the full path to the Excel spreadsheet used as a template for the results file.
% out = A string representing the full path to the Excel spreadsheet to which the results are written, eventually replacing the previous ones.
% bandwidth = An integer [21,252] representing the dimension of each rolling window (optional, default=252).
% k = A float [0.90,0.99] representing the confidence level used to calculate the CATFIN (optional, default=0.99).
% f = A float [0.2,0.8] representing the percentage of components to include in the computation of the Absorption Ratio (optional, default=0.2).
% q = A float (0.5,1.0) representing the threshold of the Turbulence Index (optional, default=0.75).
% analyze = A boolean that indicates whether to analyse the results and display plots (optional, default=false).
%
% [OUTPUT]
% result = A structure representing the original dataset inclusive of intermediate and final calculations.
% stopped = A boolean that indicates whether the process has been stopped through user input.

function [result,stopped] = run_component(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data',@(x)validateattributes(x,{'struct'},{'nonempty'}));
        ip.addRequired('temp',@(x)validateattributes(x,{'char'},{'nonempty','size',[1 NaN]}));
        ip.addRequired('out',@(x)validateattributes(x,{'char'},{'nonempty','size',[1 NaN]}));
        ip.addOptional('bandwidth',252,@(x)validateattributes(x,{'numeric'},{'scalar','integer','real','finite','>=',21,'<=',252}));
        ip.addOptional('k',0.99,@(x)validateattributes(x,{'double'},{'scalar','real','finite','>=',0.90,'<=',0.99}));
        ip.addOptional('f',0.2,@(x)validateattributes(x,{'double'},{'scalar','real','finite','>=',0.2,'<=',0.8}));
        ip.addOptional('q',0.75,@(x)validateattributes(x,{'double'},{'scalar','real','finite','>',0.5,'<',1}));
        ip.addOptional('analyze',false,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    data = validate_dataset(ipr.data,'component');
    temp = validate_template(ipr.temp);
    out = validate_output(ipr.out);
    
    nargoutchk(1,2);
    
    [result,stopped] = run_component_internal(data,temp,out,ipr.bandwidth,ipr.k,ipr.f,ipr.q,ipr.analyze);

end

function [result,stopped] = run_component_internal(data,temp,out,bandwidth,k,f,q,analyze)

    result = [];
    stopped = false;
    e = [];
    
    data = data_initialize(data,bandwidth,k,f,q);
    n = data.N;
    t = data.T;
    
    rng_settings = rng();
    rng(0);
    
    bar = waitbar(0,'Initializing component measures...','CreateCancelBtn',@(src,event)setappdata(gcbf(),'Stop',true));
    setappdata(bar,'Stop',false);
    cleanup = onCleanup(@()delete(bar));

    pause(1);
    waitbar(0,bar,'Calculating component measures...');
    pause(1);

    try

        rw = 1 ./ (repmat(n,t,1) - sum(isnan(data.Returns),2));
        data.CATFINReturns = sum(data.Returns .* repmat(rw,1,n),2,'omitnan');

        windows_fr = extract_rolling_windows(data.Returns,bandwidth,false);
        windows_pr = extract_rolling_windows(data.CATFINReturns,bandwidth,false);

        futures(1:t) = parallel.FevalFuture;
        futures_max = 0;
        futures_results = cell(t,1);

        for i = 1:t
            futures(i) = parfeval(@main_loop,1,windows_fr{i},windows_pr{i},data.Components,data.A);
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
    
    rng(rng_settings);
    
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
        data = data_finalize(data,futures_results);
    catch e
        delete(bar);
        rethrow(e);
    end

    pause(1);
    waitbar(1,bar,'Writing component measures...');
    pause(1);
    
    try
        write_results(temp,out,data);
        delete(bar);
    catch e
        delete(bar);
        rethrow(e);
    end

    if (analyze)
        safe_plot(@(id)plot_indicators_catfin(data,id));
        safe_plot(@(id)plot_indicators_other(data,id));
        safe_plot(@(id)plot_pca(data,id));
    end
    
    result = data;

end

%% DATA

function data = data_initialize(data,bandwidth,k,f,q)

    data.A = 1 - k;
    data.Bandwidth = bandwidth;
    data.Components = round(data.N * f,0);
    data.F = f;
    data.K = k;
    data.Q = q;

    k_label = sprintf('%.0f%%',(data.K * 100));
    data.Labels = {['CATFIN VaR (K=' k_label ')'] 'Indicators' 'PCA Overall Explained' 'PCA Overall Coefficients' 'PCA Overall Scores'};
    data.LabelsSimple = {'CATFIN VaR' 'Indicators' 'PCA Overall Explained' 'PCA Overall Coefficients' 'PCA Overall Scores'};
    data.LabelsVaR = {'Non-Parametric' 'GPD' 'SGED'};
    
    data.CATFINReturns = NaN(data.T,1);
    data.CATFINVaR = NaN(data.T,3);
    data.CATFINFirstCoefficients = NaN(1,3);
    data.CATFINFirstExplained = NaN;
    data.CATFIN = NaN(data.T,1);
    
    data.AbsorptionRatio = NaN(data.T,1);
    data.CorrelationSurprise = NaN(data.T,1);
    data.TurbulenceIndex = NaN(data.T,1);

    data.PCACoefficients = cell(data.T,1);
    data.PCAExplained = cell(data.T,1);
    data.PCAExplainedSums = NaN(data.T,4);
    data.PCAScores = cell(data.T,1);

end

function data = data_finalize(data,results)

    t = data.T;

    for i = 1:t
        window_result = results{i};
        
        data.CATFINVaR(i,:) = window_result.CATFINVaR;
        
        data.AbsorptionRatio(i) = window_result.AbsorptionRatio;
        data.CorrelationSurprise(i) = window_result.CorrelationSurprise;
        data.TurbulenceIndex(i) = window_result.TurbulenceIndex;
        
        data.PCACoefficients{i} = window_result.PCACoefficients;
        data.PCAExplained{i} = window_result.PCAExplained;
        data.PCAExplainedSums(i,:) = fliplr([cumsum([window_result.PCAExplained(1) window_result.PCAExplained(2) window_result.PCAExplained(3)]) 100]);
        data.PCAScores{i} = window_result.PCAScores;
    end
    
    data.AbsorptionRatio = min(max(data.AbsorptionRatio,0),1);
    
    [coefficients,scores,explained] = calculate_pca(data.CATFINVaR,false);
    data.CATFIN = scores(:,1);
    data.CATFINFirstCoefficients = coefficients(:,1).';
    data.CATFINFirstExplained = explained(1);

    r = data.Returns;
    nan_indices = isnan(r);
    rm = repmat(mean(r,1,'omitnan'),t,1);
    r(nan_indices) = rm(nan_indices);

    [coefficients,scores,explained] = calculate_pca(r,true);
    data.PCACoefficientsOverall = coefficients;
    data.PCAExplainedOverall = explained;
    data.PCAExplainedSumsOverall = fliplr([cumsum([explained(1) explained(2) explained(3)]) 100]);
    data.PCAScoresOverall = scores;

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
    
    sheets = {'CATFIN VaR' 'Indicators' 'PCA Overall Explained' 'PCA Overall Coefficients' 'PCA Overall Scores'};

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
        error('The results file could not be created from the template file.');
    end

    dates_str = cell2table(data.DatesStr,'VariableNames',{'Date'});

    labels = strrep(data.LabelsVaR,'-','_');
    t1 = [dates_str array2table(data.CATFINVaR,'VariableNames',labels)];
    writetable(t1,out,'FileType','spreadsheet','Sheet','CATFIN VaR','WriteRowNames',true);

    labels = {'Absorption_Ratio' 'CATFIN' 'Correlation_Surprise' 'Turbulence_Index'};
    t2 = [dates_str array2table([data.AbsorptionRatio data.CATFIN data.CorrelationSurprise data.TurbulenceIndex],'VariableNames',labels)];
    writetable(t2,out,'FileType','spreadsheet','Sheet','Indicators','WriteRowNames',true);

    labels = {'PC' 'ExplainedVariance'};
    t3 = array2table([(1:data.N).' data.PCAExplainedOverall],'VariableNames',labels);
    writetable(t3,out,'FileType','spreadsheet','Sheet','PCA Overall Explained','WriteRowNames',true);

    labels = [{'Firms'} data.FirmNames];
    t4 = cell2table([data.FirmNames.' num2cell(data.PCACoefficientsOverall)],'VariableNames',labels);
    writetable(t4,out,'FileType','spreadsheet','Sheet','PCA Overall Coefficients','WriteRowNames',true);

    labels = data.FirmNames;
    t5 = [dates_str array2table(data.PCAScoresOverall,'VariableNames',labels)];
    writetable(t5,out,'FileType','spreadsheet','Sheet','PCA Overall Scores','WriteRowNames',true);
    
    if (ispc())
        try
            excel = actxserver('Excel.Application');
        catch
            return;
        end

        try
            exc_wb = excel.Workbooks.Open(out,0,false);

            for i = 1:numel(data.LabelsSimple)
                exc_wb.Sheets.Item(data.LabelsSimple{i}).Name = data.Labels{i};
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

function window_results = main_loop(window_fr,window_pr,components,a)

    nan_indices = sum(isnan(window_fr),1) > 0;
    window_fr(:,nan_indices) = [];

    window_results = struct();

    [var_np,var_gpd,var_sged] = calculate_catfin_var(window_pr,a,0.98,0.05);
    window_results.CATFINVaR = -1 .* min(0,[var_np var_gpd var_sged]);
    
    [ar,cs,ti] = calculate_component_indicators(window_fr,components);
    window_results.AbsorptionRatio = ar;
    window_results.CorrelationSurprise = cs;
    window_results.TurbulenceIndex = ti;
    
    [coefficients,scores,explained] = calculate_pca(window_fr,true);
    window_results.PCACoefficients = coefficients;
    window_results.PCAExplained = explained;
    window_results.PCAScores = scores;

end

function [var_np,var_gpd,var_sged] = calculate_catfin_var(data,a,g,u)

    t = size(data,1);

    w = fliplr(((1 - g) / (1 - g^t)) .* (g .^ (0:1:t-1))).';  
    h = sortrows([data w],1,'ascend');
    csw = cumsum(h(:,2));
    cswa = find(csw >= a);
    var_np = h(cswa(1),1);  

    k = round(t / (t * u),0);
    data_neg = -data; 
    data_sneg = sort(data_neg);
    u = data_sneg(t - k);
    excess = data_neg(data_neg > u) - u;
    gpd_params = gpfit(excess);
    [xi,beta,zeta] = deal(gpd_params(1),gpd_params(2),k / t);
    var_gpd = -(u + (beta / xi) * ((((1 / zeta) * a) ^ -xi) - 1));

    try
        sged_params = mle(data,'PDF',@sgedpdf,'Start',[mean(data) std(data) 0 1],'LowerBound',[-Inf 0 -1 0],'UpperBound',[Inf Inf 1 Inf]);
        [mu,sigma,lambda,kappa] = deal(sged_params(1),sged_params(2),sged_params(3),sged_params(4));
        var_sged = fsolve(@(x)sgedcdf(x,mu,sigma,lambda,kappa)-a,0,optimset(optimset(@fsolve),'Diagnostics','off','Display','off'));
    catch
        var_sged = var_gpd;
    end

end

function [ar,cs,ti] = calculate_component_indicators(data,components)

    zero_indices = find(~data);
    data(zero_indices) = (-9e-9 .* rand(numel(zero_indices),1)) + 1e-8;
    
    novar_indices = find(var(data,1) == 0);
    data(:,novar_indices) = data(:,novar_indices) + ((-9e-9 .* rand(size(data(:,novar_indices)))) + 1e-8);

    c = cov(data);
    bm = eye(size(c)) .* diag(c);
    e = eigs(c,size(c,1));

    v = data(end,:) - mean(data(1:end-1,:),1);
    vt = v.';

    ar = sum(e(1:components)) / trace(c);
    cs = ((v / c) * vt) / ((v / bm) * vt);
    ti = (v / c) * vt;

end

function [coefficients,scores,explained] = calculate_pca(data,normalize)

    if (normalize)
        for i = 1:size(data,2)
            c = data(:,i);

            m = mean(c,'omitnan');

            s = std(c,'omitnan');
            s(s == 0) = 1;

            data(:,i) = (c - m) ./ s;
        end
    end

    [coefficients,scores,~,~,explained] = pca(data,'Economy',false);

end

function p = sgedcdf(x,mu,sigma,lambda,kappa)

    [t,n] = size(x);
    p = NaN(t,n);

    for i = 1:t
        for j = 1:n
            p(i) = integral(@(x)sgedpdf(x,mu,sigma,lambda,kappa),-Inf,x(i,j));
        end
    end

end

function y = sgedpdf(x,mu,sigma,lambda,kappa)

    g1 = gammaln(1 / kappa);
    g2 = gammaln(2 / kappa);
    g3 = gammaln(3 / kappa);

    a = exp(g2 - (0.5 * g1) - (0.5 * g3));
    s = sqrt(1 + (3 * lambda^2) - (4 * a^2 * lambda^2));

    theta = exp((0.5 * g1) - (0.5 * g3)) / s;
    delta = (2 * lambda * a) / s;

    c = exp(log(kappa) - (log(2 * sigma * theta) + g1));
    u = x - mu + (delta * sigma);

    y = c .*exp((-abs(u) .^ kappa) ./ ((1 + (sign(u) .* lambda)) .^ kappa) ./ theta^kappa ./ sigma^kappa); 

end

%% PLOTTING

function plot_indicators_catfin(data,id)

    r = max(0,-data.CATFINReturns);

    y_max = max(max([-data.CATFINVaR r]));
    y_limits = [0 ((abs(y_max) * 1.1) * sign(y_max))];

    f = figure('Name','Component Measures > CATFIN Indicator','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    sub_1 = subplot(2,3,[1 3]);
    plot(sub_1,data.DatesNum,data.CATFIN,'Color',[0.000 0.447 0.741]);
    set(sub_1,'XLim',[data.DatesNum(1) data.DatesNum(end)],'XTickLabelRotation',45);
    t1 = title(sub_1,['CATFIN (K=' sprintf('%.0f%%',data.K * 100) ', PCA.EV=' sprintf('%.2f%%',data.CATFINFirstExplained) ')']);
    set(t1,'Units','normalized');
    t1_position = get(t1,'Position');
    set(t1,'Position',[0.4783 t1_position(2) t1_position(3)]);

    if (data.MonthlyTicks)
        datetick(sub_1,'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        datetick(sub_1,'x','yyyy','KeepLimits');
    end
    
    for i = 1:3
        sub = subplot(2,3,i + 3);
        plot(sub,data.DatesNum,r,'Color',[0.000 0.447 0.741]);
        hold on;
            plot(sub,data.DatesNum,data.CATFINVaR(:,i),'Color',[1 0.4 0.4]);
        hold off;
        set(sub,'XLim',[data.DatesNum(1) data.DatesNum(end)],'XTickLabelRotation',45,'YLim',y_limits);
        t1 = title(sub,[data.LabelsVaR{i} ' Value-at-Risk (PCA.C=' sprintf('%.4f',data.CATFINFirstCoefficients(i)) ')']);
        set(t1,'Units','normalized');
        t1_position = get(t1,'Position');
        set(t1,'Position',[0.4783 t1_position(2) t1_position(3)]);

        if (data.MonthlyTicks)
            datetick(sub,'x','mm/yyyy','KeepLimits','KeepTicks');
        else
            datetick(sub,'x','yyyy','KeepLimits');
        end
    end

    t = figure_title('CATFIN Indicator');
    t_position = get(t,'Position');
    set(t,'Position',[t_position(1) -0.0157 t_position(3)]);
    
    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_indicators_other(data,id)

    alpha = 2 / (data.Bandwidth + 1);

    ti_th = NaN(data.T,1);

    for i = 1:data.T
        ti_th(i) = quantile(data.TurbulenceIndex(max(1,i-data.Bandwidth):min(data.T,i+data.Bandwidth)),data.Q);
    end

    ti_ma = [data.TurbulenceIndex(1,:); filter(alpha,[1 (alpha - 1)],data.TurbulenceIndex(2:end,:),(1 - alpha) * data.TurbulenceIndex(1,:))];
	cs_ma = [data.CorrelationSurprise(1,:); filter(alpha,[1 (alpha - 1)],data.CorrelationSurprise(2:end,:),(1 - alpha) * data.CorrelationSurprise(1,:))];

    f = figure('Name','Component Measures > Other Indicators','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    sub_1 = subplot(2,2,[1 3]);
    plot(sub_1,data.DatesNum,data.AbsorptionRatio,'Color',[0.000 0.447 0.741]);
    xlabel(sub_1,'Time');
    ylabel(sub_1,'Value');
    set(sub_1,'YLim',[0 1],'YTick',0:0.1:1,'YTickLabels',arrayfun(@(x)sprintf('%.f%%',x),(0:0.1:1) * 100,'UniformOutput',false));
    t1 = title(sub_1,['Absorption Ratio (F=' sprintf('%.2f',data.F) ')']);
    set(t1,'Units','normalized');
    t1_position = get(t1,'Position');
    set(t1,'Position',[0.4783 t1_position(2) t1_position(3)]);

    sub_2 = subplot(2,2,2);
    p2 = plot(sub_2,data.DatesNum,data.TurbulenceIndex,'Color',[0.65 0.65 0.65]);
    p2.Color(4) = 0.35;
    hold on;
        p21 = plot(sub_2,data.DatesNum,ti_ma,'Color',[0.000 0.447 0.741]);
        p22 = plot(sub_2,data.DatesNum,ti_th,'Color',[1 0.4 0.4]);
    hold off;
    xlabel(sub_2,'Time');
    ylabel(sub_2,'Value');
    l = legend(sub_2,[p21 p22],'EWMA','Threshold','Location','best');
    set(l,'NumColumns',2,'Units','normalized');
    l_position = get(l,'Position');
    set(l,'Position',[0.6710 0.4799 l_position(3) l_position(4)]);
    t2 = title(sub_2,['Turbulence Index (Q=' sprintf('%.2f',data.Q) ')']);
    set(t2,'Units','normalized');
    t2_position = get(t2,'Position');
    set(t2,'Position',[0.4783 t2_position(2) t2_position(3)]);
    
    sub_3 = subplot(2,2,4);
    p3 = plot(sub_3,data.DatesNum,data.CorrelationSurprise,'Color',[0.65 0.65 0.65]);
    p3.Color(4) = 0.35;
    hold on;
        plot(sub_3,data.DatesNum,cs_ma,'Color',[0.000 0.447 0.741]);
    hold off;
    xlabel(sub_3,'Time');
    ylabel(sub_3,'Value');
    t3 = title(sub_3,'Correlation Surprise');
    set(t3,'Units','normalized');
    t3_position = get(t3,'Position');
    set(t3,'Position',[0.4783 t3_position(2) t3_position(3)]);
    
    set([sub_1 sub_2 sub_3],'XLim',[data.DatesNum(1) data.DatesNum(end)],'XTickLabelRotation',45);

    if (data.MonthlyTicks)
        datetick(sub_1,'x','mm/yyyy','KeepLimits','KeepTicks');
        datetick(sub_2,'x','mm/yyyy','KeepLimits','KeepTicks');
        datetick(sub_3,'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        datetick(sub_1,'x','yyyy','KeepLimits');
        datetick(sub_2,'x','yyyy','KeepLimits');
        datetick(sub_3,'x','yyyy','KeepLimits');
    end

    t = figure_title('Other Indicators');
    t_position = get(t,'Position');
    set(t,'Position',[t_position(1) -0.0157 t_position(3)]);
    
    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_pca(data,id)

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
    
    f = figure('Name','Component Measures > Principal Component Analysis','Units','normalized','Tag',id);

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
    set(sub_2,'XLim',[data.DatesNum(1) data.DatesNum(end)],'XTick',[],'YLim',[y_ticks(1) y_ticks(end)],'YTick',y_ticks,'YTickLabel',y_labels);
    legend(sub_2,sprintf('PC 4-%d',data.N),'PC 3','PC 2','PC 1','Location','southeast');
    title('Explained Variance');

    t = figure_title('Principal Component Analysis');
    t_position = get(t,'Position');
    set(t,'Position',[t_position(1) -0.0157 t_position(3)]);
    
    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end
