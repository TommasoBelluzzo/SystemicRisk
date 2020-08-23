% [INPUT]
% ds = A structure representing the dataset.
% temp = A string representing the full path to the Excel spreadsheet used as a template for the results file.
% out = A string representing the full path to the Excel spreadsheet to which the results are written, eventually replacing the previous ones.
% k = A float [0.90,0.99] representing the confidence level (optional, default=0.95).
% car = A float [0.03,0.20] representing the capital adequacy ratio (optional, default=0.08).
% sf = A float [0,1] representing the fraction of separate accounts, if available, to include in liabilities (optional, default=0.40).
% d = A float [0.1,0.6] representing the six-month crisis threshold for the market index decline used to calculate the LRMES (optional, default=0.40).
% fr = An integer [0,6] representing the number of months of forward-rolling used to calculate the SRISK, which simulates the difficulty of renegotiating debt in case of financial distress (optional, default=3).
% analyze = A boolean that indicates whether to analyse the results and display plots (optional, default=false).
%
% [OUTPUT]
% result = A structure representing the original dataset inclusive of intermediate and final calculations.
% stopped = A boolean that indicates whether the process has been stopped through user input.

function [result,stopped] = run_cross_sectional(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('ds',@(x)validateattributes(x,{'struct'},{'nonempty'}));
        ip.addRequired('temp',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addRequired('out',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addOptional('k',0.95,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.90 '<=' 0.99 'scalar'}));
        ip.addOptional('car',0.08,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.03 '<=' 0.20 'scalar'}));
        ip.addOptional('sf',0.40,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0 '<=' 1 'scalar'}));
        ip.addOptional('d',0.40,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.1 '<=' 0.6 'scalar'}));
        ip.addOptional('fr',3,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 0 '<=' 6 'scalar'}));
        ip.addOptional('analyze',false,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    ds = validate_dataset(ipr.ds,'cross-sectional');
    temp = validate_template(ipr.temp);
    out = validate_output(ipr.out);
    k = ipr.k;
    car = ipr.car;
    sf = ipr.sf;
    d = ipr.d;
    fr = ipr.fr;
    analyze = ipr.analyze;
    
    nargoutchk(1,2);

    [result,stopped] = run_cross_sectional_internal(ds,temp,out,k,car,sf,d,fr,analyze);

end

function [result,stopped] = run_cross_sectional_internal(ds,temp,out,k,car,sf,d,fr,analyze)

    result = [];
    stopped = false;
    e = [];

    ds = initialize(ds,k,car,sf,d,fr);
    n = ds.N;
    t = ds.T;

    bar = waitbar(0,'Initializing cross-sectional measures...','CreateCancelBtn',@(src,event)setappdata(gcbf(),'Stop', true));
    setappdata(bar,'Stop',false);
    cleanup = onCleanup(@()delete(bar));
    
    pause(1);
    waitbar(0,bar,'Calculating cross-sectional measures...');
    pause(1);

    try

        r_m = ds.Index - mean(ds.Index);
        r_x = ds.Returns;

        cp = ds.Capitalizations;

        lb = ds.Liabilities;
        lbr = apply_forward_rolling(ds.Liabilities,ds.DatesNum,ds.FR);
        
        sa = ds.SeparateAccounts;
        
        if (~isempty(sa))
            lb = lb - ((1 - ds.SF) .* sa);
            
            sar = apply_forward_rolling(ds.SeparateAccounts,ds.DatesNum,ds.FR);
            lbr = lbr - ((1 - ds.SF) .* sar);
        end

        sv = ds.StateVariables;

        for i = 1:n
            waitbar((i - 1) / n,bar,['Calculating cross-sectional measures for ' ds.FirmNames{i} '...']);

            if (getappdata(bar,'Stop'))
                stopped = true;
                break;
            end
            
            offset = min(ds.Defaults(i) - 1,t);

            r0_m = r_m(1:offset)- mean(r_m(1:offset));
            r0_x = r_x(1:offset,i) - mean(r_x(1:offset,i));

            cp_x = cp(1:offset,i);

            lb_x = lb(1:offset,i);
            lbr_x = lbr(1:offset,i);
            
            if (isempty(sv))
                sv_x = [];
            else
                sv_x = sv(1:offset,:);
            end

            [p,h] = dcc_gjrgarch([r0_m r0_x]);
            s_m = sqrt(h(:,1));
            s_x = sqrt(h(:,2));
            rho = squeeze(p(1,2,:));

            [beta,var,es] = calculate_idiosyncratic(s_m,r0_x,s_x,rho,ds.A);
            [covar,dcovar] = calculate_covar(r0_m,r0_x,var,sv_x,ds.A);
            [mes,lrmes] = calculate_mes(r0_m,s_m,r0_x,s_x,rho,beta,ds.A,ds.D);
            ses = calculate_ses(lb_x,cp_x,ds.CAR);
            srisk = calculate_srisk(lbr_x,cp_x,lrmes,ds.CAR);

            ds.Beta(1:offset,i) = beta;
            ds.VaR(1:offset,i) = -1 * var;
            ds.ES(1:offset,i) = -1 * es;
            ds.CoVaR(1:offset,i) = -1 * covar;
            ds.DeltaCoVaR(1:offset,i) = -1 * dcovar;
            ds.MES(1:offset,i) = -1 * mes;
            ds.SES(1:offset,i) = ses;
            ds.SRISK(1:offset,i) = srisk;
            
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
        ds = finalize(ds);
    catch e
        delete(bar);
        rethrow(e);
    end
    
    pause(1);
    waitbar(1,bar,'Writing cross-sectional measures...');
    pause(1);
    
    try
        write_results(ds,temp,out);
        delete(bar);
    catch e
        delete(bar);
        rethrow(e);
    end
    
    if (analyze)
        safe_plot(@(id)plot_idiosyncratic_averages(ds,id));
        safe_plot(@(id)plot_sequence(ds,'Beta',id));
        safe_plot(@(id)plot_sequence(ds,'VaR',id));
        safe_plot(@(id)plot_sequence(ds,'ES',id));
        safe_plot(@(id)plot_systemic_averages(ds,id));
        safe_plot(@(id)plot_sequence(ds,'CoVaR',id));
        safe_plot(@(id)plot_sequence(ds,'Delta CoVaR',id));
        safe_plot(@(id)plot_sequence(ds,'MES',id));
        safe_plot(@(id)plot_sequence(ds,'SES',id));
        safe_plot(@(id)plot_sequence(ds,'SRISK',id));
        safe_plot(@(id)plot_correlations(ds,id));
        safe_plot(@(id)plot_rankings(ds,id));
    end
    
    result = ds;

end

%% DATA

function ds = initialize(ds,k,car,sf,d,fr)

    n = ds.N;
    t = ds.T;

    ds.A = 1 - k;
    ds.CAR = car;
    ds.D = d;
    ds.FR = fr;
    ds.K = k;
    ds.SF = sf;

    car_label = sprintf('%.1f%%',(ds.CAR * 100));
    d_label = sprintf('%.1f%%',(ds.D * 100));
    k_label = sprintf('%.1f%%',(ds.K * 100));
    
    k_all_label = [' (K=' k_label ')'];
    ses_label =  [' (CAR=' car_label ')'];
    srisk_label = [' (D=' d_label ', CAR=' car_label ')'];

    ds.LabelsMeasuresSimple = {'Beta' 'VaR' 'ES' 'CoVaR' 'Delta CoVaR' 'MES' 'SES' 'SRISK'};
    ds.LabelsMeasures = {'Beta' ['VaR' k_all_label] ['ES' k_all_label] ['CoVaR' k_all_label] ['Delta CoVaR' k_all_label] ['MES' k_all_label] ['SES' ses_label] ['SRISK' srisk_label]};

    ds.LabelsSheetsSimple = [ds.LabelsMeasuresSimple {'Averages'}];
    ds.LabelsSheets = [ds.LabelsMeasures {'Averages'}];

    ds.Beta = NaN(t,n);
    ds.VaR = NaN(t,n);
    ds.ES = NaN(t,n);
    ds.CoVaR = NaN(t,n);
    ds.DeltaCoVaR = NaN(t,n);
    ds.MES = NaN(t,n);
    ds.SES = NaN(t,n);
    ds.SRISK = NaN(t,n);
    ds.Averages = NaN(t,8);

    ds.RankingConcordance = NaN(8,8);
    ds.RankingStability = NaN(1,8);
    
    ds.ComparisonReferences = {'Averages' 4:8 strcat({'CS-'},strrep(ds.LabelsMeasuresSimple(4:end),'Delta ','D'))};

end

function ds = finalize(ds)

    n = ds.N;

    weights = max(0,ds.Capitalizations ./ repmat(sum(ds.Capitalizations,2,'omitnan'),1,n));
    
    beta_avg = sum(ds.Beta .* weights,2,'omitnan');
    var_avg = sum(ds.VaR .* weights,2,'omitnan');
    es_avg = sum(ds.ES .* weights,2,'omitnan');
    covar_avg = sum(ds.CoVaR .* weights,2,'omitnan');
    dcovar_avg = sum(ds.DeltaCoVaR .* weights,2,'omitnan');
    mes_avg = sum(ds.MES .* weights,2,'omitnan');
    ses_avg = sum(ds.SES .* weights,2,'omitnan');
    srisk_avg = sum(ds.SRISK .* weights,2,'omitnan');
    ds.Averages = [beta_avg var_avg es_avg covar_avg dcovar_avg mes_avg ses_avg srisk_avg];

    [rc,rs] = kendall_rankings(ds,ds.LabelsMeasuresSimple);
    ds.RankingConcordance = rc;
    ds.RankingStability = rs;

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

    sheets = {'Beta' 'VaR' 'ES' 'CoVaR' 'Delta CoVaR' 'MES' 'SES' 'SRISK' 'Averages'};
    
    if (~all(ismember(sheets,file_sheets)))
        error(['The template must contain the following sheets: ' sheets{1} sprintf(', %s',sheets{2:end}) '.']);
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

    dates_str = cell2table(ds.DatesStr,'VariableNames',{'Date'});

    for i = 1:(numel(ds.LabelsSheetsSimple) - 1)
        sheet = ds.LabelsSheetsSimple{i};
        measure = strrep(sheet,' ','');

        tab = [dates_str array2table(ds.(measure),'VariableNames',ds.FirmNames)];
        writetable(tab,out,'FileType','spreadsheet','Sheet',sheet,'WriteRowNames',true);
    end

    tab = [dates_str array2table(ds.Averages,'VariableNames',strrep(ds.LabelsSheetsSimple(1:end-1),' ','_'))];
    writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{end},'WriteRowNames',true);    

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

function data_fr = apply_forward_rolling(data,dates_num,fr)

    if (~isempty(data) && (fr > 0))
        [~,a] = unique(cellstr(datestr(dates_num,'mm/yyyy')),'stable');
        data_monthly = data(a,:);

        indices_seq = [a(1:fr:numel(a)) - 1; numel(dates_num)];
        data_seq = data_monthly(1:fr:numel(a),:);

        data_fr = NaN(size(data));

        for i = 2:numel(indices_seq)
            indices = (indices_seq(i-1) + 1):indices_seq(i);
            data_fr(indices,:) = repmat(data_seq(i-1,:),numel(indices),1);
        end
    else
        data_fr = data;
    end

end

function [covar,dcovar] = calculate_covar(r0_m,r0_x,var,sv,a)

    if (isempty(sv))
        b = quantile_regression(r0_m,r0_x,a);
        covar = b(1) + (b(2) .* var);
    else
        b = quantile_regression(r0_m(2:end),[r0_x(2:end) sv(1:end-1,:)],a);
        covar = b(1) + (b(2) .* var(2:end));

        for i = 1:size(sv,2)
            covar = covar + (b(i+2) .* sv(1:end-1,i));
        end
        
        covar = [covar(1); covar];
    end

    dcovar = b(2) .* (var - repmat(median(r0_x),length(r0_m),1));

end

function [beta,var,es] = calculate_idiosyncratic(s_m,r0_x,s_x,rho,a)

    beta = rho .* (s_x ./ s_m);
    
    c = quantile((r0_x ./ s_x),a);
    var = s_x * c;
    es = s_x * -(normpdf(c) / a);

end

function [mes,lrmes] = calculate_mes(r0_m,s_m,r0_x,s_x,rho,beta,a,d)

    c = quantile(r0_m,a);
    z = sqrt(1 - rho.^2);

    u = r0_m ./ s_m;
    x = ((r0_x ./ s_x) - (rho .* u)) ./ z;

    r0_n = 4 / (3 * length(r0_m));
    r0_s = min([std(r0_m ./ s_m) (iqr(r0_m ./ s_m) ./ 1.349)]);
    h = r0_s * r0_n ^0.2;

    f = normcdf(((c ./ s_m) - u) ./ h);
    f_sum = sum(f);

    k1 = sum(u .* f) ./ f_sum;
    k2 = sum(x .* f) ./ f_sum;

    mes = (s_x .* rho .* k1) + (s_x .* z .* k2);
    lrmes = 1 - exp(log(1 - d) .* beta);

end

function ses = calculate_ses(lb,eq,z)

    lb_pc = [0; diff(lb) ./ lb(1:end-1)];
    eq_pc = [0; diff(eq) ./ eq(1:end-1)];
    
    ses = (z .* lb .* (1 + lb_pc)) - ((1 - z) .* eq .* (1 + eq_pc));
    ses(ses < 0) = 0;

end

function srisk = calculate_srisk(lb,eq,lrmes,car)

    srisk = (car .* lb) - ((1 - car) .* (1 - lrmes) .* eq);
    srisk(srisk < 0) = 0;

end

function beta = quantile_regression(y,x,k)

    [n,m] = size(x);
    m = m + 1;

    x = [ones(n,1) x];
    x_star = x;

    beta = ones(m,1);

    diff = 1;
    i = 0;

    while ((diff > 1e-6) && (i < 1000))
        x_star_t = x_star.';
        beta_0 = beta;

        beta = ((x_star_t * x) \ x_star_t) * y;

        residuals = y - (x * beta);
        residuals(abs(residuals) < 1e-06) = 1e-06;
        residuals(residuals < 0) = k * residuals(residuals < 0);
        residuals(residuals > 0) = (1 - k) * residuals(residuals > 0);
        residuals = abs(residuals);

        z = zeros(n,m);

        for j = 1:m 
            z(:,j) = x(:,j) ./ residuals;
        end

        x_star = z;
        beta_1 = beta;
        
        diff = max(abs(beta_1 - beta_0));
        i = i + 1;
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

function plot_idiosyncratic_averages(ds,id)

    averages = ds.Averages(:,1:3);
    beta = averages(:,1);
    others = averages(:,2:3);

    y_limits_beta = plot_limits(beta,0.1,0);
    y_limits_others = plot_limits(others,0.1);

    f = figure('Name','Cross-Sectional Measures > Idiosyncratic Averages','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);
    
    sub_1 = subplot(2,2,[1 3]);
    plot(sub_1,ds.DatesNum,smooth_data(beta),'Color',[0.000 0.447 0.741]);
    set(sub_1,'YLim',y_limits_beta);
    title(sub_1,ds.LabelsMeasures{1});
    
    sub_2 = subplot(2,2,2);
    plot(sub_2,ds.DatesNum,smooth_data(averages(:,2)),'Color',[0.000 0.447 0.741]);
    set(sub_2,'YLim',y_limits_others);
    title(sub_2,ds.LabelsMeasures{2});
    
    sub_3 = subplot(2,2,4);
    plot(sub_3,ds.DatesNum,smooth_data(averages(:,3)),'Color',[0.000 0.447 0.741]);
    set(sub_3,'YLim',y_limits_others,'YTick',get(sub_2,'YTick'),'YTickLabel',get(sub_2,'YTickLabel'),'YTickLabelMode',get(sub_2,'YTickLabelMode'),'YTickMode',get(sub_2,'YTickMode'));
    title(sub_3,ds.LabelsMeasures{3});
    
    set([sub_1 sub_2 sub_3],'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
    set([sub_1 sub_2 sub_3],'XGrid','on','YGrid','on');

    if (ds.MonthlyTicks)
        date_ticks([sub_1 sub_2 sub_3],'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        date_ticks([sub_1 sub_2 sub_3],'x','yyyy','KeepLimits');
    end

    figure_title('Idiosyncratic Averages');

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_systemic_averages(ds,id)

    y_limits = zeros(5,2);

    averages_quantile = ds.Averages(:,4:6);
    y_limits(1:3,:) = repmat(plot_limits(averages_quantile,0.1),3,1);
    
    averages_volume = ds.Averages(:,7:8);
    y_limits(4:5,:) = repmat(plot_limits(averages_volume,0.1),2,1);
    
    subplot_offsets = cell(5,1);
    subplot_offsets{1} = [1 3 5];
    subplot_offsets{2} = [7 9 11];
    subplot_offsets{3} = [13 15 17];
    subplot_offsets{4} = [2 4 6 8];
    subplot_offsets{5} = [12 14 16 18];

    f = figure('Name','Cross-Sectional Measures > Systemic Averages','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);
    
    subs = gobjects(5,1);
    height_delta = NaN;
    
    for i = 1:5
        sub = subplot(9,2,subplot_offsets{i});
        plot(sub,ds.DatesNum,smooth_data(ds.Averages(:,i+3)),'Color',[0.000 0.447 0.741]);
        set(sub,'YLim',y_limits(i,:));
        title(sub,ds.LabelsMeasures{i+3});

        if (i == 1)
            sub_position = get(sub,'Position');
            height_old = sub_position(4);
            height_new = height_old * 0.8;
            height_delta = height_old - height_new;
            
            set(sub,'Position',[sub_position(1) (sub_position(2) + height_delta) sub_position(3) (sub_position(4) - height_delta)]);
        else
            sub_position = get(sub,'Position');
            set(sub,'Position',[sub_position(1) (sub_position(2) + height_delta) sub_position(3) (sub_position(4) - height_delta)]);
        end
        
        subs(i) = sub;
    end
    
    set(subs,'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
    set(subs,'XGrid','on','YGrid','on');
    
    if (ds.MonthlyTicks)
        date_ticks(subs,'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        date_ticks(subs,'x','yyyy','KeepLimits');
    end
    
    y_ticks = get(subs(1),'YTick');
    y_tick_labels = arrayfun(@(x)sprintf('%.2f',x),y_ticks,'UniformOutput',false);
    set(subs(1:3),'YTick',y_ticks,'YTickLabel',y_tick_labels);
    
    y_ticks = get(subs(4),'YTick');
    y_tick_labels = arrayfun(@(x)sprintf('%.0f',x),y_ticks,'UniformOutput',false);
    set(subs(4:5),'YTick',y_ticks,'YTickLabel',y_tick_labels);
    
    figure_title('Systemic Averages');

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_correlations(ds,id)

    mu = mean(ds.Averages,1);
    sigma = std(ds.Averages,1);
    
    [rho,pval] = corr(ds.Averages);
    rho(isnan(rho)) = 0;

    z = bsxfun(@minus,ds.Averages,mu);
    z = bsxfun(@rdivide,z,sigma);
    z_limits = [nanmin(z(:)) nanmax(z(:))];
    
    n = numel(ds.LabelsMeasures);

    f = figure('Name','Cross-Sectional Measures > Correlation Matrix','Units','normalized','Tag',id);
    
    [ax,big_ax] = gplotmatrix_stable(f,ds.Averages,ds.LabelsMeasuresSimple);

    x_labels = get(ax,'XLabel');
    y_labels = get(ax,'YLabel');
    set([x_labels{:}; y_labels{:}],'FontWeight','bold');
    
    x_labels_grey = arrayfun(@(l)l{1},x_labels);
    x_labels_grey_indices = ismember({x_labels_grey.String},ds.LabelsMeasuresSimple(1:3));
    y_labels_grey = arrayfun(@(l)l{1},y_labels);
    y_labels_grey_indices = ismember({y_labels_grey.String},ds.LabelsMeasuresSimple(1:3));
    set([x_labels{x_labels_grey_indices}; y_labels{y_labels_grey_indices}],'Color',[0.5 0.5 0.5]);

    for i = 1:n
        for j = 1:n
            ax_ij = ax(i,j);
            
            z_limits_current = 1.1 .* z_limits;
            x_limits = mu(j) + (z_limits_current * sigma(j));
            y_limits = mu(i) + (z_limits_current * sigma(i));

            set(get(big_ax,'Parent'),'CurrentAxes',ax_ij);
            set(ax_ij,'XLim',x_limits,'XTick',[]);
            set(ax_ij,'YLim',y_limits,'YTick',[]);
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

function plot_rankings(ds,id)

    labels = ds.LabelsMeasuresSimple;
    n = numel(labels);
    seq = 1:n;
    off = seq + 0.5;

    [rs,order] = sort(ds.RankingStability);
    rs_names = labels(order);
    
    rc = ds.RankingConcordance;
    rc(rc <= 0.5) = 0;
    rc(rc > 0.5) = 1;
    rc(logical(eye(n))) = 0.5;
    
    [rc_x,rc_y] = meshgrid(seq,seq);
    rc_x = rc_x(:) + 0.5;
    rc_y = rc_y(:) + 0.5;
    rc_text = cellstr(num2str(ds.RankingConcordance(:),'%.2f'));

    f = figure('Name','Cross-Sectional Measures > Rankings','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    sub_1 = subplot(1,2,1);
    bar(sub_1,seq,rs,'FaceColor',[0.749 0.862 0.933]);
    set(sub_1,'XTickLabel',rs_names);
    set(sub_1,'YLim',[0 1]);
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
    colormap([1 1 1; 0.65 0.65 0.65; 0.749 0.862 0.933]);
    axis('image');
    text(rc_x,rc_y,rc_text,'FontSize',9,'HorizontalAlignment','center');
    set(sub_2,'FontWeight','bold','TickLength',[0 0]);
    set(sub_2,'XAxisLocation','bottom','XTick',off,'XTickLabels',labels,'XTickLabelRotation',45);
    set(sub_2,'YDir','reverse','YTick',off,'YTickLabels',labels,'YTickLabelRotation',45)
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
    
    figure_title('Rankings (Kendall''s W)');

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_sequence(ds,target,id)

    n = ds.N;
    t = ds.T;
    dn = ds.DatesNum;
    mt = ds.MonthlyTicks;

    ts = smooth_data(ds.(strrep(target,' ','')));

    data = [repmat({dn},1,n); mat2cell(ts,t,ones(1,n))];
    
    [~,index] = ismember(target,ds.LabelsMeasuresSimple);
    plots_title = repmat(ds.LabelsMeasures(index),1,n);
    
    x_limits = [dn(1) dn(end)];
    y_limits = plot_limits(ts,0.1);

    core = struct();

    core.N = n;
    core.Data = data;
    core.Function = @(subs,data)plot_function(subs,data);

    core.OuterTitle = ['Cross-Sectional Measures > ' target ' Time Series'];
    core.InnerTitle = [target ' Time Series'];
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
    core.YLimits = {y_limits};
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
