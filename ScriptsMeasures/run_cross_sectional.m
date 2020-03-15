% [INPUT]
% data = A structure representing the dataset.
% temp = A string representing the full path to the Excel spreadsheet used as a template for the results file.
% out = A string representing the full path to the Excel spreadsheet to which the results are written, eventually replacing the previous ones.
% k = A float [0.90,0.99] representing the confidence level used to calculate CoVaR, Delta CoVaR, MES and LRMES (optional, default=0.95).
% d = A float [0.1,0.6] representing the six-month crisis threshold for the market index decline used to calculate the LRMES (optional, default=0.40).
% car = A float [0.03,0.20] representing the capital adequacy ratio used to calculate the SRISK (optional, default=0.08).
% sf = A float [0,1] representing the fraction of separate accounts, if available, to include in liabilities during the SRISK calculation (optional, default=0.40).
% analyze = A boolean that indicates whether to analyse the results and display plots (optional, default=false).
%
% [OUTPUT]
% result = A structure representing the original dataset inclusive of intermediate and final calculations.
% stopped = A boolean that indicates whether the process has been stopped through user input.

function [result,stopped] = run_cross_sectional(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data',@(x)validateattributes(x,{'struct'},{'nonempty'}));
        ip.addRequired('temp',@(x)validateattributes(x,{'char'},{'nonempty','size',[1,NaN]}));
        ip.addRequired('out',@(x)validateattributes(x,{'char'},{'nonempty','size',[1,NaN]}));
        ip.addOptional('k',0.95,@(x)validateattributes(x,{'double'},{'scalar','real','finite','>=',0.90,'<=',0.99}));
        ip.addOptional('d',0.40,@(x)validateattributes(x,{'double'},{'scalar','real','finite','>=',0.1,'<=',0.6}));
        ip.addOptional('car',0.08,@(x)validateattributes(x,{'double'},{'scalar','real','finite','>=',0.03,'<=',0.20}));
        ip.addOptional('sf',0.40,@(x)validateattributes(x,{'double'},{'scalar','real','finite','>=',0,'<=',1}));
        ip.addOptional('analyze',false,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    data = validate_dataset(ipr.data,'cross-sectional');
    temp = validate_template(ipr.temp);
    out = validate_output(ipr.out);
    
    nargoutchk(1,2);

    [result,stopped] = run_cross_sectional_internal(data,temp,out,ipr.k,ipr.d,ipr.car,ipr.sf,ipr.analyze);

end

function [result,stopped] = run_cross_sectional_internal(data,temp,out,k,d,car,sf,analyze)

    result = [];
    stopped = false;
    e = [];

    data = data_initialize(data,k,d,car,sf);
    n = data.N;

    bar = waitbar(0,'Initializing cross-sectional measures...','CreateCancelBtn',@(src,event)setappdata(gcbf(),'Stop', true));
    setappdata(bar,'Stop',false);
    
    pause(1);
    waitbar(0,bar,'Calculating cross-sectional measures...');
    pause(1);

    try

        r0_m = data.IndexReturns - mean(data.IndexReturns);
        
        eq = data.Capitalization;
        lb = data.Liabilities;
        
        if (isempty(data.SeparateAccounts))
            sa = zeros(0,n);
        else
            sa = data.SeparateAccounts;
        end
        
        sv = data.StateVariables;
        
        for i = 1:n
            waitbar((i - 1) / n,bar,['Calculating cross-sectional measures for ' data.FirmNames{i} '...']);

            if (getappdata(bar,'Stop'))
                stopped = true;
                break;
            end
            
            r_x = data.FirmReturns(:,i);
            r0_x = r_x - mean(r_x);

            [p,h] = dcc_gjrgarch([r0_m r0_x]);
            s_m = sqrt(h(:,1));
            s_x = sqrt(h(:,2));
            rho = squeeze(p(1,2,:));

            [beta,var,es] = calculate_idiosyncratic(data.A,s_m,r0_x,s_x,rho);
            [covar,dcovar] = calculate_covar(data.A,r0_m,r0_x,var,sv);
            [mes,lrmes] = calculate_mes(data.A,data.D,r0_m,s_m,r0_x,s_x,rho,beta);
            srisk = calculate_srisk(data.CAR,data.SF,lrmes,lb(:,i),eq(:,i),sa(:,i));

            data.Beta(:,i) = beta;
            data.VaR(:,i) = -1 * var;
            data.ES(:,i) = -1 * es;
            data.CoVaR(:,i) = -1 * covar;
            data.DeltaCoVaR(:,i) = -1 * dcovar;
            data.MES(:,i) = -1 * mes;
            data.SRISK(:,i) = srisk;
            
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
        data = data_finalize(data);
    catch e
        delete(bar);
        rethrow(e);
    end
    
    pause(1);
    waitbar(1,bar,'Writing cross-sectional measures...');
    pause(1);
    
    try
        write_results(temp,out,data);
        delete(bar);
    catch e
        delete(bar);
        rethrow(e);
    end
    
    if (analyze)
        try
            plot_idiosyncratic_averages(data);
        catch
            warning('MATLAB:SystemicRisk','The analysis function ''plot_idiosyncratic_averages'' produced errors.');
        end
        
        try
            plot_systemic_averages(data);
        catch
            warning('MATLAB:SystemicRisk','The analysis function ''plot_systemic_averages'' produced errors.');
        end
        
        try
            plot_correlations(data);
        catch
            warning('MATLAB:SystemicRisk','The analysis function ''plot_correlations'' produced errors.');
        end
        
        try
            plot_rankings(data);
        catch
            warning('MATLAB:SystemicRisk','The analysis function ''plot_rankings'' produced errors.');
        end
    end
    
    result = data;

end

%% DATA

function data = data_initialize(data,k,d,car,sf)
  
    data.A = 1 - k;
    data.CAR = car;
    data.D = d;
    data.K = k;
    data.SF = sf;

    car_label = sprintf('%.0f%%',(data.CAR * 100));
    d_label = sprintf('%.0f%%',(data.D * 100));
    k_label = sprintf('%.0f%%',(data.K * 100));
    data.Labels = {'Beta' ['VaR (K=' k_label ')'] ['ES (K=' k_label ')'] ['CoVaR (K=' k_label ')'] ['Delta CoVaR (K=' k_label ')'] ['MES (K=' k_label ')'] ['SRISK (D=' d_label ', CAR=' car_label ')'] 'Averages'};
    data.LabelsSimple = {'Beta' 'VaR' 'ES' 'CoVaR' 'Delta CoVaR' 'MES' 'SRISK' 'Averages'};
    
    data.Beta = NaN(data.T,data.N);
    data.VaR = NaN(data.T,data.N);
    data.ES = NaN(data.T,data.N);
    data.CoVaR = NaN(data.T,data.N);
    data.DeltaCoVaR = NaN(data.T,data.N);
    data.MES = NaN(data.T,data.N);
    data.SRISK = NaN(data.T,data.N);

end

function data = data_finalize(data)

    factors = sum(data.Capitalization,2);
    weights = data.CapitalizationLagged ./ repmat(sum(data.CapitalizationLagged,2),1,data.N);

    data.Beta = handle_defaulted_firms(data.Beta,data.FirmDefaults);
    beta_avg = sum(data.Beta .* weights,2,'omitnan');
    
    data.VaR = handle_defaulted_firms(data.VaR,data.FirmDefaults);
    var_avg = sum(data.VaR .* weights,2,'omitnan') .* factors;

    data.ES = handle_defaulted_firms(data.ES,data.FirmDefaults);
    es_avg = sum(data.ES .* weights,2,'omitnan') .* factors;
    
    data.CoVaR = handle_defaulted_firms(data.CoVaR,data.FirmDefaults);
    covar_avg = sum(data.CoVaR .* weights,2,'omitnan') .* factors;
    
    data.DeltaCoVa = handle_defaulted_firms(data.DeltaCoVaR,data.FirmDefaults);
    dcovar_avg = sum(data.DeltaCoVa .* weights,2,'omitnan') .* factors;

    data.MES = handle_defaulted_firms(data.MES,data.FirmDefaults);
    mes_avg = sum(data.MES .* weights,2,'omitnan') .* factors;
    
    data.SRISK = handle_defaulted_firms(data.SRISK,data.FirmDefaults);
    srisk_avg = sum(data.SRISK .* weights,2,'omitnan');

    data.Averages = [beta_avg var_avg es_avg covar_avg dcovar_avg mes_avg srisk_avg];
    
    measures = numel(data.LabelsSimple) - 1;
    measures_pairs = nchoosek(1:measures,2);
    
    data.RankingConcordance = zeros(measures,measures);
    data.RankingStability = zeros(1,measures);
    
    for i = 1:size(measures_pairs,1)
        pair = measures_pairs(i,:);

        index_1 = pair(1);
        field_1 = strrep(data.LabelsSimple{index_1},' ','');
        measure_1 = data.(field_1);
        
        index_2 = pair(2);
        field_2 = strrep(data.LabelsSimple{index_2},' ','');
        measure_2 = data.(field_2);
        
        for j = 1:data.T
            [~,rank_1] = sort(measure_1(j,:));
            [~,rank_2] = sort(measure_2(j,:));

            data.RankingConcordance(index_1,index_2) = data.RankingConcordance(index_1,index_2) + kendall_concordance_coefficient(rank_1.',rank_2.');
        end
    end
    
    for i = 1:measures
        field = strrep(data.LabelsSimple{i},' ','');
        measure = data.(field);
        
        for j = data.T:-1:2
            [~,rank_previous] = sort(measure(j-1,:));
            [~,rank_current] = sort(measure(j,:));

            data.RankingStability(i) = data.RankingStability(i) + kendall_concordance_coefficient(rank_current.',rank_previous.');
        end
    end
    
    data.RankingConcordance = ((data.RankingConcordance + data.RankingConcordance.') / data.T) + eye(measures);
    data.RankingStability = data.RankingStability ./ (data.T - 1);

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
            error('The template file is not a valid Excel spreadsheet.');
        end
    else
        [file_status,file_sheets] = xlsfinfo(temp);
        
        if (isempty(file_status))
            error('The template file is not a valid Excel spreadsheet.');
        end
    end

    sheets = {'Beta' 'VaR' 'ES' 'CoVaR' 'Delta CoVaR' 'MES' 'SRISK' 'Averages'};
    
    if (~all(ismember(sheets,file_sheets)))
        error(['The template must contain the following sheets: ' sheets{1} sprintf(', %s', sheets{2:end}) '.']);
    end
    
    if (ispc())
        try
            excel = actxserver('Excel.Application');
            excel_wb = excel.Workbooks.Open(temp,0,false);

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

    dates_str = cell2table(data.DatesStr,'VariableNames',{'Date'});

    for i = 1:(numel(data.LabelsSimple) - 1)
        sheet = data.LabelsSimple{i};
        measure = strrep(sheet,' ','');

        tab = [dates_str array2table(data.(measure),'VariableNames',data.FirmNames)];
        writetable(tab,out,'FileType','spreadsheet','Sheet',sheet,'WriteRowNames',true);
    end

    tab = [dates_str array2table(data.Averages,'VariableNames',strrep(data.LabelsSimple(1:end-1),' ','_'))];
    writetable(tab,out,'FileType','spreadsheet','Sheet','Averages','WriteRowNames',true);    

    if (ispc())
        try
            excel = actxserver('Excel.Application');
            exc_wb = excel.Workbooks.Open(out,0,false);

            for i = 1:numel(data.LabelsSimple)
                exc_wb.Sheets.Item(data.LabelsSimple{i}).Name = data.Labels{i};
            end
            
            exc_wb.Save();
            exc_wb.Close();
            excel.Quit();
            
            delete(excel);
        catch
        end
    end

end

%% MEASURES

function [covar,dcovar] = calculate_covar(a,r0_m,r0_x,var,sv)

    if (isempty(sv))
        b = quantile_regression(r0_m,r0_x,a);
        covar = b(1) + (b(2) .* var);
    else
        b = quantile_regression(r0_m,[r0_x sv],a);
        covar = b(1) + (b(2) .* var);

        for i = 1:size(sv,2)
            covar = covar + (b(i+2) .* sv(:,i));
        end
    end

	dcovar = b(2) .* (var - repmat(median(r0_x),length(r0_m),1));

end

function [beta,var,es] = calculate_idiosyncratic(a,s_m,r0_x,s_x,rho)

	beta = rho .* (s_x ./ s_m);
    
    c = quantile((r0_x ./ s_x),a);
	var = s_x * c;
	es = s_x * -(normpdf(c) / a);

end

function [mes,lrmes] = calculate_mes(a,d,r0_m,s_m,r0_x,s_x,rho,beta)

    c = quantile(r0_m,a);
    z = sqrt(1 - rho.^2);

    u = r0_m ./ s_m;
    x = ((r0_x ./ s_x) - (rho .* u)) ./ z;

    r0_n = 4 / (3 * length(r0_m));
    r0_s = min([std(r0_m) (iqr(r0_m) / 1.349)]);
    h = r0_s * (r0_n ^ (-0.2));

    f = normcdf(((c ./ s_m) - u) ./ h);
    f_sum = sum(f);

    k1 = sum(u .* f) ./ f_sum;
    k2 = sum(x .* f) ./ f_sum;

    mes = (s_x .* rho .* k1) + (s_x .* z .* k2);
    lrmes = 1 - exp(log(1 - d) .* beta);

end

function srisk = calculate_srisk(l,s,lrmes,lb,eq,sa)

    if (~isempty(sa))
        lb = lb - ((1 - s) .* sa);
    end

    srisk = (l .* lb) - ((1 - l) .* (1 - lrmes) .* eq);
    srisk(srisk < 0) = 0;

end

function kcc = kendall_concordance_coefficient(rank_1,rank_2)

	m = [rank_1 rank_2];
	[n,k] = size(m);

    rm = zeros(n,k);

    for i = 1:k
        x_i = m(:,i);
        [~,b] = sortrows(x_i);
        rm(b,i) = 1:n;
    end

    rm_sum = sum(rm,2);
    s = sum(rm_sum.^2,1) - ((sum(rm_sum) ^ 2) / n);

    kcc = (12 * s) / ((k ^ 2) * (( n^ 3) - n));

end

function beta = quantile_regression(y,x,k)

    [n,m] = size(x);
    m = m + 1;

    x = [ones(n,1) x];
    x_star = x;

    beta = ones(m,1);

    diff = 1;
    iter = 0;

    while ((diff > 1e-6) && (iter < 1000))
        x_star_t = x_star.';
        beta_0 = beta;

        beta = ((x_star_t * x) \ x_star_t) * y;

        residuals = y - (x * beta);
        residuals(abs(residuals) < 0.000001) = 0.000001;
        residuals(residuals < 0) = k * residuals(residuals < 0);
        residuals(residuals > 0) = (1 - k) * residuals(residuals > 0);
        residuals = abs(residuals);

        z = zeros(n,m);

        for i = 1:m 
            z(:,i) = x(:,i) ./ residuals;
        end

        x_star = z;
        beta_1 = beta;
        
        diff = max(abs(beta_1 - beta_0));
        iter = iter + 1;
    end

end

%% PLOTTING


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

function plot_idiosyncratic_averages(data)

    averages = data.Averages(:,1:3);
    
    y_min = min(min(averages));
    y_max = max(max(averages(:,1)));
    y_limits_beta = [((abs(y_min) * 1.1) * sign(y_min)) ((abs(y_max) * 1.1) * sign(y_max))];

    y_min = min(min(averages));
    y_max = max(max(averages(:,2:3)));
    y_limits_other = [((abs(y_min) * 1.1) * sign(y_min)) ((abs(y_max) * 1.1) * sign(y_max))];
    
    y_limits = [y_limits_beta; y_limits_other; y_limits_other];

    f = figure('Name','Cross-Sectional Measures > Idiosyncratic Averages','Units','normalized','Position',[100 100 0.85 0.85]);
    
    subs = NaN(3,1);

    for i = 1:3
        sub = subplot(1,3,i);
        plot(sub,data.DatesNum,averages(:,i));
        xlabel(sub,'Time');
        ylabel(sub,'Value');
        set(sub,'XLim',[data.DatesNum(1) data.DatesNum(end)],'YLim',y_limits(i,:),'XTickLabelRotation',45);
        title(sub,data.Labels(i));
        
        if (data.MonthlyTicks)
            datetick(sub,'x','mm/yyyy','KeepLimits','KeepTicks');
        else
            datetick(sub,'x','yyyy','KeepLimits');
        end
        
        subs(i) = sub;
    end

    y_labels = arrayfun(@(x)sprintf('%.0f',x),get(subs(end),'YTick'),'UniformOutput',false);
    set([subs(2) subs(3)],'YTickLabel',y_labels);
    
    t = figure_title('Idiosyncratic Averages');
    t_position = get(t,'Position');
    set(t,'Position',[t_position(1) -0.0157 t_position(3)]);

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_systemic_averages(data)

    averages = data.Averages(:,4:end);

    y_min = min(min(averages));
    y_max = max(max(averages));
    y_limits = [((abs(y_min) * 1.1) * sign(y_min)) ((abs(y_max) * 1.1) * sign(y_max))];

    f = figure('Name','Cross-Sectional Measures > Systemic Averages','Units','normalized','Position',[100 100 0.85 0.85]);

    subs = NaN(4,1);
    
    for i = 1:4
        sub = subplot(2,2,i);
        plot(sub,data.DatesNum,averages(:,i));
        xlabel(sub,'Time');
        ylabel(sub,'Value');
        set(sub,'XLim',[data.DatesNum(1) data.DatesNum(end)],'YLim',y_limits,'XTickLabelRotation',45);
        title(sub,data.Labels(i+3));
        
        if (data.MonthlyTicks)
            datetick(sub,'x','mm/yyyy','KeepLimits','KeepTicks');
        else
            datetick(sub,'x','yyyy','KeepLimits');
        end
        
        subs(i) = sub;
    end

    y_labels = arrayfun(@(x)sprintf('%.0f',x),get(subs(end),'YTick'),'UniformOutput',false);
    set(subs,'YTickLabel',y_labels);
    
    t = figure_title('Systemic Averages');
    t_position = get(t,'Position');
    set(t,'Position',[t_position(1) -0.0157 t_position(3)]);

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_correlations(data)

    [rho,pval] = corr(data.Averages);
    m = mean(data.Averages,1);
    s = std(data.Averages,1);

    z = bsxfun(@minus,data.Averages,m);
    z = bsxfun(@rdivide,z,s);
    z_limits = [nanmin(z(:)) nanmax(z(:))];
    
    n = numel(data.LabelsSimple) - 1;

    f = figure('Name','Cross-Sectional Measures > Correlation Matrix','Units','normalized');
    
    [ax,big_ax] = gplotmatrix_stable(f,data.Averages,data.LabelsSimple(1:end-1));

    x_labels = get(ax,'XLabel');
    y_labels = get(ax,'YLabel');
    set([x_labels{:}; y_labels{:}],'FontWeight','bold');
    
    x_labels_grey = arrayfun(@(l)l{1},x_labels);
    x_labels_grey_indices = ismember({x_labels_grey.String},data.LabelsSimple(1:3));
    y_labels_grey = arrayfun(@(l)l{1},y_labels);
    y_labels_grey_indices = ismember({y_labels_grey.String},data.LabelsSimple(1:3));
    set([x_labels{x_labels_grey_indices}; y_labels{y_labels_grey_indices}],'Color',[0.5 0.5 0.5]);

    for i = 1:n
        for j = 1:n
            ax_ij = ax(i,j);
            
            z_limits_current = 1.1 .* z_limits;
            x_limits = m(j) + (z_limits_current * s(j));
            y_limits = m(i) + (z_limits_current * s(i));

            set(get(big_ax,'Parent'),'CurrentAxes',ax_ij);
            set(ax_ij,'XLim',x_limits,'XTick',[],'YLim',y_limits,'YTick',[]);
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

function plot_rankings(data)

    labels = data.LabelsSimple(1:end-1);
    n = numel(labels);
    seq = 1:n;
    off = seq + 0.5;

    [rs,order] = sort(data.RankingStability);
    rs_names = labels(order);
    
    rc = data.RankingConcordance;
    rc(rc <= 0.5) = 0.0;
    rc(rc > 0.5) = 1.0;
    rc(logical(eye(n))) = 0.5;
    
    [rc_x,rc_y] = meshgrid(seq,seq);
    rc_x = rc_x(:) + 0.5;
    rc_y = rc_y(:) + 0.5;
    rc_text = cellstr(num2str(data.RankingConcordance(:),'%.2f'));

    f = figure('Name','Cross-Sectional Measures > Rankings','Units','normalized','Position',[100 100 0.85 0.85]);

    sub_1 = subplot(1,2,1);
    bar(sub_1,seq,rs,'FaceColor',[0.749 0.862 0.933]);
    set(sub_1,'XTickLabel',rs_names,'YLim',[0 1]);
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
    colormap([1 1 1; 0.65 0.65 0.65; 0.749 0.862 0.933])
    axis image;
    text(rc_x, rc_y, rc_text,'FontSize',9,'HorizontalAlignment','center');
    set(sub_2,'FontWeight','bold','XAxisLocation','bottom','TickLength',[0 0],'YDir','reverse');
    set(sub_2,'XTick',off,'XTickLabels',labels,'XTickLabelRotation',45,'YTick',off,'YTickLabels',labels,'YTickLabelRotation',45)
    t2 = title(sub_2,'Ranking Concordance');
    t2_position = get(t2,'Position');
    set(t2,'Position',[t2_position(1) 0.2897 t2_position(3)]);

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
    
    t = figure_title('Rankings (Kendall''s W)');
    t_position = get(t,'Position');
    set(t,'Position',[t_position(1) -0.0157 t_position(3)]);

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end
