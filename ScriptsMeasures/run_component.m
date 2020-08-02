% [INPUT]
% ds = A structure representing the dataset.
% temp = A string representing the full path to the Excel spreadsheet used as a template for the results file.
% out = A string representing the full path to the Excel spreadsheet to which the results are written, eventually replacing the previous ones.
% bw = An integer [21,252] representing the dimension of each rolling window (optional, default=252).
% k = A float [0.90,0.99] representing the confidence level used to calculate the CATFIN (optional, default=0.99).
% f = A float [0.2,0.8] representing the percentage of components to include in the computation of the Absorption Ratio (optional, default=0.2).
% q = A float (0.5,1.0) representing the quantile threshold of the Turbulence Index (optional, default=0.75).
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
        ip.addRequired('temp',@(x)validateattributes(x,{'char'},{'nonempty','size',[1 NaN]}));
        ip.addRequired('out',@(x)validateattributes(x,{'char'},{'nonempty','size',[1 NaN]}));
        ip.addOptional('bw',252,@(x)validateattributes(x,{'double'},{'real','finite','integer','>=',21,'<=',252,'scalar'}));
        ip.addOptional('k',0.99,@(x)validateattributes(x,{'double'},{'real','finite','>=',0.90,'<=',0.99,'scalar'}));
        ip.addOptional('f',0.2,@(x)validateattributes(x,{'double'},{'real','finite','>=',0.2,'<=',0.8,'scalar'}));
        ip.addOptional('q',0.75,@(x)validateattributes(x,{'double'},{'real','finite','>',0.5,'<',1,'scalar'}));
        ip.addOptional('analyze',false,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    ds = validate_dataset(ipr.ds,'component');
    temp = validate_template(ipr.temp);
    out = validate_output(ipr.out);
    bw = ipr.bw;
    k = ipr.k;
    f = ipr.f;
    q = ipr.q;
    analyze = ipr.analyze;
    
    nargoutchk(1,2);
    
    [result,stopped] = run_component_internal(ds,temp,out,bw,k,f,q,analyze);

end

function [result,stopped] = run_component_internal(ds,temp,out,bw,k,f,q,analyze)

    result = [];
    stopped = false;
    e = [];
    
    ds = initialize(ds,bw,k,f,q);
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
        windows_rc = extract_rolling_windows(ds.CATFINReturns,ds.BW);

        futures(1:t) = parallel.FevalFuture;
        futures_max = 0;
        futures_results = cell(t,1);

        for i = 1:t
            futures(i) = parfeval(@main_loop,1,windows_rf{i},windows_rc{i},ds.Components,ds.A);
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
        safe_plot(@(id)plot_catfin(ds,id));
        safe_plot(@(id)plot_indicators_other(ds,id));
        safe_plot(@(id)plot_pca(ds,id));
    end
    
    result = ds;

end

%% DATA

function ds = initialize(ds,bw,k,f,q)

    n = ds.N;
    t = ds.T;

    r = ds.Returns;
    rw = 1 ./ (repmat(n,t,1) - sum(isnan(r),2));

    ds.A = 1 - k;
    ds.BW = bw;
    ds.Components = round(ds.N * f,0);
    ds.F = f;
    ds.K = k;
    ds.Q = q;

    k_label = sprintf('%.0f%%',(ds.K * 100));
    ds.Labels = {['CATFIN VaR (K=' k_label ')'] 'Indicators' 'PCA Overall Explained' 'PCA Overall Coefficients' 'PCA Overall Scores'};
    ds.LabelsCATFINVaR = {'Non-Parametric' 'GPD' 'GEV' 'SGED'};
    ds.LabelsIndicators = {'Absorption Ratio' 'CATFIN' 'Correlation Surprise' 'Turbulence Index'};
    ds.LabelsPCAExplained = {'PC' 'Explained Variance'};
    ds.LabelsSimple = {'CATFIN VaR' 'Indicators' 'PCA Overall Explained' 'PCA Overall Coefficients' 'PCA Overall Scores'};

    ds.CATFINReturns = sum(r .* repmat(rw,1,n),2,'omitnan');
    ds.CATFINVaR = NaN(t,4);
    ds.CATFINFirstCoefficients = NaN(1,3);
    ds.CATFINFirstExplained = NaN;
    ds.CATFIN = NaN(t,1);
    
    ds.AbsorptionRatio = NaN(t,1);
    ds.CorrelationSurprise = NaN(t,1);
    ds.TurbulenceIndex = NaN(t,1);

    ds.PCACoefficients = cell(t,1);
    ds.PCAExplained = cell(t,1);
    ds.PCAExplainedSums = NaN(t,4);
    ds.PCAScores = cell(t,1);
    
    ds.PCACoefficientsOverall = NaN(n,n);
    ds.PCAExplainedOverall = NaN(n,1);
    ds.PCAExplainedSumsOverall = NaN(1,4);
    ds.PCAScoresOverall = NaN(t,n);

end

function ds = finalize(ds,results)

    t = ds.T;

    for i = 1:t
        window_result = results{i};
        
        ds.CATFINVaR(i,:) = window_result.CATFINVaR;
        
        ds.AbsorptionRatio(i) = window_result.AbsorptionRatio;
        ds.CorrelationSurprise(i) = window_result.CorrelationSurprise;
        ds.TurbulenceIndex(i) = window_result.TurbulenceIndex;
        
        ds.PCACoefficients{i} = window_result.PCACoefficients;
        ds.PCAExplained{i} = window_result.PCAExplained;
        ds.PCAExplainedSums(i,:) = fliplr([cumsum([window_result.PCAExplained(1) window_result.PCAExplained(2) window_result.PCAExplained(3)]) 100]);
        ds.PCAScores{i} = window_result.PCAScores;
    end

    ds.CATFINVaR(:,4) = sanitize_data(ds.CATFINVaR(:,4),ds.DatesNum,[],[]);

    [coefficients,scores,explained] = calculate_pca(ds.CATFINVaR,false);
    ds.CATFIN = scores(:,1);
    ds.CATFINFirstCoefficients = coefficients(:,1).';
    ds.CATFINFirstExplained = explained(1);

    ds.AbsorptionRatio = sanitize_data(ds.AbsorptionRatio,ds.DatesNum,[],[0 1]);

    [coefficients,scores,explained] = calculate_overall_pca(ds.Returns);
    ds.PCACoefficientsOverall = coefficients;
    ds.PCAExplainedOverall = explained;
    ds.PCAExplainedSumsOverall = fliplr([cumsum([explained(1) explained(2) explained(3)]) 100]);
    ds.PCAScoresOverall = scores;

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

    labels = strrep(ds.LabelsCATFINVaR,'-','_');
    t1 = [dates_str array2table(ds.CATFINVaR,'VariableNames',labels)];
    writetable(t1,out,'FileType','spreadsheet','Sheet','CATFIN VaR','WriteRowNames',true);

    labels = strrep(ds.LabelsIndicators,' ','_');
    t2 = [dates_str array2table([ds.AbsorptionRatio ds.CATFIN ds.CorrelationSurprise ds.TurbulenceIndex],'VariableNames',labels)];
    writetable(t2,out,'FileType','spreadsheet','Sheet','Indicators','WriteRowNames',true);

    labels = strrep(ds.LabelsPCAExplained,' ','_');
    t3 = array2table([(1:ds.N).' ds.PCAExplainedOverall],'VariableNames',labels);
    writetable(t3,out,'FileType','spreadsheet','Sheet','PCA Overall Explained','WriteRowNames',true);

    labels = [{'Firms'} ds.FirmNames];
    t4 = cell2table([ds.FirmNames.' num2cell(ds.PCACoefficientsOverall)],'VariableNames',labels);
    writetable(t4,out,'FileType','spreadsheet','Sheet','PCA Overall Coefficients','WriteRowNames',true);

    labels = ds.FirmNames;
    t5 = [dates_str array2table(ds.PCAScoresOverall,'VariableNames',labels)];
    writetable(t5,out,'FileType','spreadsheet','Sheet','PCA Overall Scores','WriteRowNames',true);
    
    if (ispc())
        try
            excel = actxserver('Excel.Application');
        catch
            return;
        end

        try
            exc_wb = excel.Workbooks.Open(out,0,false);

            for i = 1:numel(ds.LabelsSimple)
                exc_wb.Sheets.Item(ds.LabelsSimple{i}).Name = ds.Labels{i};
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

function window_results = main_loop(rf,rc,components,a)

    window_results = struct();

    nan_indices = sum(isnan(rf),1) > 0;
    rf(:,nan_indices) = [];

    [var_np,var_gpd,var_gev,var_sged] = calculate_catfin_var(rc,a,0.98,0.05);
    window_results.CATFINVaR = -1 .* min(0,[var_np var_gpd var_gev var_sged]);
    
    [ar,cs,ti] = calculate_component_indicators(rf,components);
    window_results.AbsorptionRatio = ar;
    window_results.CorrelationSurprise = cs;
    window_results.TurbulenceIndex = ti;
    
    [coefficients,scores,explained] = calculate_pca(rf,true);
    window_results.PCACoefficients = coefficients;
    window_results.PCAExplained = explained;
    window_results.PCAScores = scores;

end

function [var_np,var_gpd,var_gev,var_sged] = calculate_catfin_var(x,a,g,u)

    persistent options;

    if (isempty(options))
        options = optimset(optimset(@fsolve),'Diagnostics','off','Display','off');
    end

    t = size(x,1);

    w = fliplr(((1 - g) / (1 - g^t)) .* (g .^ (0:1:t-1))).';  
    h = sortrows([x w],1,'ascend');
    csw = cumsum(h(:,2));
    cswa = find(csw >= a);
    var_np = h(cswa(1),1);  

    k = round(t / (t * u),0);
    x_neg = -x; 
    x_neg_sorted = sort(x_neg);
    threshold = x_neg_sorted(t - k);
    excess = x_neg(x_neg > threshold) - threshold;
    gpd_params = gpfit(excess);
    [xi,beta,zeta] = deal(gpd_params(1),gpd_params(2),k / t);
    var_gpd = -(threshold + (beta / xi) * ((((1 / zeta) * a) ^ -xi) - 1));

    k = round(nthroot(t,1.81),0);
    block_maxima = find_block_maxima(x,t,k);
    theta = find_extremal_index(x,t,k);
    gev_params = gevfit(block_maxima);
    [xi,sigma,mu] = deal(gev_params(1),gev_params(2),gev_params(3));
    var_gev = -(mu - (sigma / xi) * (1 - (-(t / k) * theta * log(1 - a))^-xi));

    try
        sged_params = mle(x,'PDF',@sgedpdf,'Start',[mean(x) std(x) 0 1],'LowerBound',[-Inf 0 -1 0],'UpperBound',[Inf Inf 1 Inf]);
        [mu,sigma,lambda,kappa] = deal(sged_params(1),sged_params(2),sged_params(3),sged_params(4));
        var_sged = fsolve(@(x)sgedcdf(x,mu,sigma,lambda,kappa)-a,0,options);
    catch
        var_sged = NaN;
    end

end

function [ar,cs,ti] = calculate_component_indicators(x,components)

    zero_indices = find(~x);
    x(zero_indices) = (-9e-9 .* rand(numel(zero_indices),1)) + 1e-8;
    
    novar_indices = find(var(x,1) == 0);
    x(:,novar_indices) = x(:,novar_indices) + ((-9e-9 .* rand(size(x(:,novar_indices)))) + 1e-8);

    c = cov(x);
    bm = eye(size(c)) .* diag(c);
    e = eigs(c,size(c,1));

    v = x(end,:) - mean(x(1:end-1,:),1);
    vt = v.';

    ar = sum(e(1:components)) / trace(c);
    cs = ((v / c) * vt) / ((v / bm) * vt);
    ti = (v / c) * vt;

end

function [coefficients,scores,explained] = calculate_overall_pca(r)

    nan_indices = isnan(r);
    rm = repmat(mean(r,1,'omitnan'),size(r,1),1);
    r(nan_indices) = rm(nan_indices);

    [coefficients,scores,explained] = calculate_pca(r,true);

end

function [coefficients,scores,explained] = calculate_pca(x,normalize)

    if (normalize)
        for i = 1:size(x,2)
            c = x(:,i);

            m = mean(c,'omitnan');

            s = std(c,'omitnan');
            s(s == 0) = 1;

            x(:,i) = (c - m) ./ s;
        end
    end

    [coefficients,scores,~,~,explained] = pca(x,'Economy',false);

end

function block_maxima = find_block_maxima(x,t,k)

    c = floor(t / k);

    block_maxima = zeros(k,1);
    i = 1;

    for j = 1:k-1
        block_maxima(j) = max(x(i:i+c-1));
        i = i + c;
    end

    block_maxima(k) = max(x(i:end));

end

function theta = find_extremal_index(x,t,k)

    c = t - k + 1;
    y = zeros(c,1);

    for i = 1:c
        y(i,1) = (1 / t) * sum(x <= max(x(i:i+k-1)));
    end

    theta = ((1 / c) * sum(-k * log(y)))^-1;

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

    y = c .* exp((-abs(u) .^ kappa) ./ ((1 + (sign(u) .* lambda)) .^ kappa) ./ theta^kappa ./ sigma^kappa); 

end

%% PLOTTING

function plot_catfin(ds,id)

    r = max(0,-ds.CATFINReturns);
    y_limits = plot_limits([-ds.CATFINVaR r],0.1,0);

    f = figure('Name','Component Measures > CATFIN','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    subs = gobjects(5,1);
    
    sub_1 = subplot(2,4,[1 4]);
    plot(sub_1,ds.DatesNum,smooth_data(ds.CATFIN),'Color',[0.000 0.447 0.741]);
    set(sub_1,'XGrid','on','YGrid','on');
    t1 = title(sub_1,['CATFIN (K=' sprintf('%.0f%%',ds.K * 100) ', PCA.EV=' sprintf('%.2f%%',ds.CATFINFirstExplained) ')']);
    set(t1,'Units','normalized');
    t1_position = get(t1,'Position');
    set(t1,'Position',[0.4783 t1_position(2) t1_position(3)]);
    
    subs(1) = sub_1;
    
    for i = 1:4
        sub = subplot(2,4,i + 4);
        plot(sub,ds.DatesNum,r,'Color',[0.000 0.447 0.741]);
        hold on;
            plot(sub,ds.DatesNum,smooth_data(ds.CATFINVaR(:,i),5),'Color',[1 0.4 0.4],'LineWidth',1.5);
        hold off;
        set(sub,'YLim',y_limits);
        title(sub,[ds.LabelsCATFINVaR{i} ' VaR (PCA.C=' sprintf('%.4f',ds.CATFINFirstCoefficients(i)) ')']);
        
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
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_indicators_other(ds,id)

    alpha = 2 / (ds.BW + 1);
    
    ar = ds.AbsorptionRatio;
    ar_limit = fix(min(ar) * 10) / 10;

    ti_th = NaN(ds.T,1);

    for i = 1:ds.T
        ti_th(i) = quantile(ds.TurbulenceIndex(max(1,i-ds.BW):min(ds.T,i+ds.BW)),ds.Q);
    end

    ti_ma = [ds.TurbulenceIndex(1); filter(alpha,[1 (alpha - 1)],ds.TurbulenceIndex(2:end),(1 - alpha) * ds.TurbulenceIndex(1))];
    cs_ma = [ds.CorrelationSurprise(1); filter(alpha,[1 (alpha - 1)],ds.CorrelationSurprise(2:end),(1 - alpha) * ds.CorrelationSurprise(1))];

    f = figure('Name','Component Measures > Other Indicators','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    sub_1 = subplot(2,2,[1 3]);
    plot(sub_1,ds.DatesNum,smooth_data(ds.AbsorptionRatio),'Color',[0.000 0.447 0.741]);
    set(sub_1,'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
    set(sub_1,'YLim',[ar_limit 1],'YTick',0:0.1:1,'YTickLabels',arrayfun(@(x)sprintf('%.f%%',x),(ar_limit:0.1:1) .* 100,'UniformOutput',false));
    set(sub_1,'XGrid','on','YGrid','on');
    t1 = title(sub_1,['Absorption Ratio (F=' sprintf('%.2f',ds.F) ')']);
    set(t1,'Units','normalized');
    t1_position = get(t1,'Position');
    set(t1,'Position',[0.4783 t1_position(2) t1_position(3)]);

    sub_2 = subplot(2,2,2);
    p2 = plot(sub_2,ds.DatesNum,ds.TurbulenceIndex,'Color',[0.65 0.65 0.65]);
    p2.Color(4) = 0.35;
    hold on;
        p21 = plot(sub_2,ds.DatesNum,ti_ma,'Color',[0.000 0.447 0.741]);
        p22 = plot(sub_2,ds.DatesNum,ti_th,'Color',[1 0.4 0.4]);
    hold off;
    set(sub_2,'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
    l = legend(sub_2,[p21 p22],'EWMA','Threshold','Location','best');
    set(l,'NumColumns',2,'Units','normalized');
    l_position = get(l,'Position');
    set(l,'Position',[0.6710 0.4895 l_position(3) l_position(4)]);
    t2 = title(sub_2,['Turbulence Index (Q=' sprintf('%.2f',ds.Q) ')']);
    set(t2,'Units','normalized');
    t2_position = get(t2,'Position');
    set(t2,'Position',[0.4783 t2_position(2) t2_position(3)]);
    
    sub_3 = subplot(2,2,4);
    p3 = plot(sub_3,ds.DatesNum,ds.CorrelationSurprise,'Color',[0.65 0.65 0.65]);
    p3.Color(4) = 0.35;
    hold on;
        plot(sub_3,ds.DatesNum,cs_ma,'Color',[0.000 0.447 0.741]);
    hold off;
    set(sub_3,'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
    t3 = title(sub_3,'Correlation Surprise');
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
    frame = get(f,'JavaFrame');
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
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end
