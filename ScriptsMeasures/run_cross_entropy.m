% [INPUT]
% ds = A structure representing the dataset.
% temp = A string representing the full path to the Excel spreadsheet used as a template for the results file.
% out = A string representing the full path to the Excel spreadsheet to which the results are written, eventually replacing the previous ones.
% bw = An integer [21,252] representing the dimension of each rolling window (optional, default=252).
% sel = A string representing the time series selection method (optional, default='F'):
%   - 'F' for firms.
%   - 'G' for groups.
% rr = A float [0,1] representing the recovery rate in case of default (optional, default=0.4).
% pw = A string representing the probabilities of default weighting method (optional, default='W'):
%   - 'A' for plain average.
%   - 'W' for progressive average.
% md = A string representing the multivariate distribution used by the CIMDO model (optional, default='N'):
%   - 'N' for normal distribution.
%   - 'T' for Student's T distribution.
% analyze = A boolean that indicates whether to analyse the results and display plots (optional, default=false).
%
% [OUTPUT]
% result = A structure representing the original dataset inclusive of intermediate and final calculations.
% stopped = A boolean that indicates whether the process has been stopped through user input.
%
% [NOTES]
% If the number of time series is greater than 10, systemic risk measures are calculated as the average of the results of multiple reduced portfolios.
% Each reduced portfolio contains 6 time series and they are created by ranking firms or groups according to the following criteria:
%   1) top 3 and bottom 3 entities by CDS spreads;
%   2) top 3 and bottom 3 entities by variance of returns;
%   3) top 3 and bottom 3 entities by market capitalization (if data is available);
%   4) top 3 and bottom 3 entities by liabilities (if data is available);

function [result,stopped] = run_cross_entropy(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('ds',@(x)validateattributes(x,{'struct'},{'nonempty'}));
        ip.addRequired('temp',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addRequired('out',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addOptional('bw',252,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 21 '<=' 252 'scalar'}));
        ip.addOptional('sel','F',@(x)any(validatestring(x,{'F' 'G'})));
        ip.addOptional('rr',0.4,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0 '<=' 1 'scalar'}));
        ip.addOptional('pw','W',@(x)any(validatestring(x,{'A' 'W'})));
        ip.addOptional('md','N',@(x)any(validatestring(x,{'N' 'T'})));
        ip.addOptional('analyze',false,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    ds = validate_dataset(ipr.ds,'cross-entropy');
    temp = validate_template(ipr.temp);
    out = validate_output(ipr.out);
    bw = ipr.bw;
    sel = validate_selection(ipr.sel,ds.Groups);
    rr = ipr.rr;
    pw = ipr.pw;
    md = ipr.md;
    analyze = ipr.analyze;
    
    nargoutchk(1,2);
    
    [result,stopped] = run_cross_entropy_internal(ds,temp,out,bw,sel,rr,pw,md,analyze);

end

function [result,stopped] = run_cross_entropy_internal(ds,temp,out,bw,sel,rr,pw,md,analyze)

    result = [];
    stopped = false;
    e = [];
    
    ds = initialize(ds,bw,sel,rr,pw,md);
    t = ds.T;
    
    bar = waitbar(0,'Initializing cross-entropy measures...','CreateCancelBtn',@(src,event)setappdata(gcbf(),'Stop',true));
    setappdata(bar,'Stop',false);
    cleanup = onCleanup(@()delete(bar));

    pause(1);
    waitbar(0,bar,'Calculating cross-entropy measures...');
    pause(1);

    try
        
        k = size(ds.Portfolios,1);
        tk = t * k;
        
        futures(1:t,1:k) = parallel.FevalFuture;
        futures_max = 0;
        futures_results = cell(t,k);

        for j = 1:k
            windows_r = extract_rolling_windows(ds.Portfolios{j,2},ds.BW);
            windows_pods = extract_rolling_windows(ds.Portfolios{j,3},ds.BW);
            
            for i = 1:t
                futures(i,j) = parfeval(@main_loop,1,windows_r{i},windows_pods{i},ds.PW,ds.MD);
            end
        end

        for i = 1:tk
            if (getappdata(bar,'Stop'))
                stopped = true;
                break;
            end
            
            [future_index,value] = fetchNext(futures);
            [future_i,future_j] = ind2sub([t k],future_index);
            futures_results{future_i,future_j} = value;
            
            futures_max = max([future_index futures_max]);
            waitbar((futures_max - 1) / tk,bar);

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
    waitbar(1,bar,'Finalizing cross-entropy measures...');
    pause(1);

    try
        ds = finalize(ds,futures_results);
    catch e
        delete(bar);
        rethrow(e);
    end

    pause(1);
    waitbar(1,bar,'Writing cross-entropy measures...');
    pause(1);
    
    try
        write_results(ds,temp,out);
        delete(bar);
    catch e
        delete(bar);
        rethrow(e);
    end

    if (analyze)
        if (size(ds.Portfolios,1) > 1)
            safe_plot(@(id)plot_portfolios_coverage(ds,id));
        end

        safe_plot(@(id)plot_indicators(ds,id));
        safe_plot(@(id)plot_dide(ds,id));
        safe_plot(@(id)plot_sequence_dide(ds,id));
        safe_plot(@(id)plot_sequence_cojpods(ds,id));
    end
    
    result = ds;

end

%% PROCESS

function ds = initialize(ds,bw,sel,rr,pw,md)

    n = ds.N;
    t = ds.T;
    
    if (strcmp(sel,'G'))
        g = ds.Groups;
        gd = ds.GroupDelimiters;
        gs = cell(g,1);
        
        for i = 1:g
            if (i == 1)
                gs{i} = 1:gd(1);
            elseif (i == g)
                gs{i} = (gd(i - 1) + 1):n;
            else
                gs{i} = (gd(i - 1) + 1):gd(i);
            end
        end

        cds_ref = ds.CDS;
        cds = zeros(t,g);
        
        r_ref = ds.Returns;
        r = zeros(t,g);

        for i = 1:g
            gs_i = gs{i};
            gs_i_len = numel(gs_i);

            r_i = r_ref(:,gs_i);
            w_i = 1 ./ (repmat(gs_i_len,t,1) - sum(isnan(r_i),2));
            r(:,i) = sum(r_i .* repmat(w_i,1,gs_i_len),2,'omitnan');

            cds_i = cds_ref(:,gs_i);
            w_i = 1 ./ (repmat(gs_i_len,t,1) - sum(isnan(cds_i),2));
            cds(:,i) = sum(cds_i .* repmat(w_i,1,gs_i_len),2,'omitnan');
        end
        
        cp_ref = ds.Capitalizations;
        
        if (isempty(cp_ref))
            cp = [];
        else
            cp = zeros(t,g);

            for i = 1:g
                cp(:,i) = sum(cp_ref(:,gs{i}),2,'omitnan');
            end
        end
        
        lb_ref = ds.Liabilities;
        
        if (isempty(lb_ref))
            lb = [];
        else
            lb = zeros(t,g);

            for i = 1:g
                lb(:,i) = sum(lb_ref(:,gs{i}),2,'omitnan');
            end
        end

        n = g;
    else
        cds = ds.CDS;
        cp = ds.Capitalizations;
        lb = ds.Liabilities;
        r = ds.Returns;
    end
    
    pods = cds ./ rr;

    if (n <= 10)
        nc = n;
        
        if (strcmp(sel,'G'))
            pfc = ds.GroupShortNames.';
        else
            pfc = ds.FirmNames;
        end
        
        pf = {'Unique' r pods []};
    else
        nc = 6;
        nch = nc / 2;

        pfc = arrayfun(@(x)sprintf('C%d',x),1:nc,'UniformOutput',false);
        pf = [repmat({''},4,1) cell(4,3)];

        rw = extract_rolling_windows(ds.Returns,bw);
        
        pf_indices = zeros(t,nc);
        pf_pods = zeros(t,nc);
        pf_r = zeros(t,nc);

        for i = 1:t
            [cds_i,indices] = sort(cds(i,:),'ascend');

            indices(isnan(cds_i)) = [];
            indices = [fliplr(indices(end-nch+1:end)) fliplr(indices(1:nch))];

            pf_r(i,:) = r(i,indices);
            pf_pods(i,:) = pods(i,indices);
            pf_indices(i,:) = indices;
        end
        
        pf(1,1:4) = {'CDS Spreads' pf_r pf_pods pf_indices};

        pf_indices = zeros(t,nc);
        pf_pods = zeros(t,nc);
        pf_r = zeros(t,nc);

        for i = 1:t
            [v_i,indices] = sort(var(rw{i}),'ascend');

            indices(isnan(v_i)) = [];
            indices = [fliplr(indices(end-nch+1:end)) fliplr(indices(1:nch))];

            pf_r(i,:) = r(i,indices);
            pf_pods(i,:) = pods(i,indices);
            pf_indices(i,:) = indices;
        end
        
        pf(2,1:4) = {'Returns Variance' pf_r pf_pods pf_indices};

        if (isempty(cp))
            offset = 3;
        else
            pf_indices = zeros(t,nc);
            pf_pods = zeros(t,nc);
            pf_r = zeros(t,nc);

            for i = 1:t
                [cp_i,indices] = sort(cp(i,:),'ascend');
                
                indices(isnan(cp_i)) = [];
                indices = [fliplr(indices(end-nch+1:end)) fliplr(indices(1:nch))];

                pf_r(i,:) = r(i,indices);
                pf_pods(i,:) = pods(i,indices);
                pf_indices(i,:) = indices;
            end
            
            pf(3,:) = {'Capitalization' pf_r pf_pods pf_indices};
            offset = 4;
        end

        if (~isempty(lb))
            pf_indices = zeros(t,nc);
            pf_pods = zeros(t,nc);
            pf_r = zeros(t,nc);

            for i = 1:t
                [lb_i,indices] = sort(lb(i,:),'ascend');
                
                indices(isnan(lb_i)) = [];
                indices = [fliplr(indices(end-nch+1:end)) fliplr(indices(1:nch))];

                pf_r(i,:) = r(i,indices);
                pf_pods(i,:) = pods(i,indices);
                pf_indices(i,:) = indices;
            end
            
            pf(offset,:) = {'Liabilities' pf_r pf_pods pf_indices};
            offset = offset + 1;
        end
        
        pf(:,offset:end) = [];
    end

    ds.BW = bw;
    ds.LGD = 1 - rr;
    ds.MD = md;
    ds.PW = pw;
    ds.RR = rr;
    ds.SEL = sel;

    ds.PortfolioComponents = pfc;
    ds.Portfolios = pf;

    all_label = [' (RR=' num2str(ds.RR * 100) '%, ' ds.PW ', ' ds.MD ')'];

    ds.LabelsMeasuresSimple = {'SI' 'SV' 'CoJPoDs'};
    ds.LabelsMeasures = {['SI' all_label] ['SV' all_label] ['CoJPoDs' all_label]};
    
    ds.LabelsIndicatorsSimple = {'JPoD' 'FSI' 'PCE'};
    ds.LabelsIndicators = {['JPoD' all_label] ['FSI' all_label] ['PCE' all_label]};

    ds.LabelsSheetsSimple = [{'Indicators' 'Average DiDe'} ds.LabelsMeasuresSimple];
    ds.LabelsSheets = [{'Indicators' 'Average DiDe'} ds.LabelsMeasures];

    ds.Indicators = NaN(t,numel(ds.LabelsIndicators));
    
    ds.AverageDiDe = NaN(nc);
    ds.DiDe = cell(t,1);
    ds.SI = NaN(t,nc);
    ds.SV = NaN(t,nc);
    
    ds.CoJPoDs = NaN(t,nc);
    
    ds.ComparisonReferences = {'Indicators' [] strcat({'CE-'},ds.LabelsIndicatorsSimple)};

end

function window_results = main_loop(r,pods,pw,md)

    window_results = struct();

    nan_indices = any(isnan(r),1);
    n = numel(nan_indices);
    
    r(:,nan_indices) = [];
    pods(:,nan_indices) = [];

    if (strcmp(pw,'A'))
        pods = mean(pods,1).';
    else
        [t,n] = size(pods);
        w = repmat(fliplr(((1 - 0.98) / (1 - 0.98^t)) .* (0.98 .^ (0:1:t-1))).',1,n);
        pods = sum(pods .* w,1).';
    end

	[g,p] = cimdo(r,pods,md);

    if (any(isnan(p)))
        window_results.JPoD = NaN;
        window_results.FSI = NaN;
        window_results.PCE = NaN;
        window_results.DiDe = NaN(n);
        window_results.SI = NaN(1,n);
        window_results.SV = NaN(1,n);
        window_results.CoJPoDs = NaN(1,n);
    else
        opods = pods;
        pods = NaN(n,1);
        pods(~nan_indices) = opods;

        [jpod,fsi,pce,dide,si,sv,cojpods] = cross_entropy_metrics(pods,g,p);
        window_results.JPoD = jpod;
        window_results.FSI = fsi;
        window_results.PCE = pce;
        window_results.DiDe = dide;
        window_results.SI = si;
        window_results.SV = sv;
        window_results.CoJPoDs = cojpods;
    end

end

function ds = finalize(ds,results)

    t = ds.T;
    k = size(results,2);

    if (k == 1)
        for i = 1:t
            result = results{i};

            ds.Indicators(i,:) = [result.JPoD result.FSI result.PCE];

            ds.DiDe{i} = result.DiDe;
            ds.SI(i,:) = result.SI;
            ds.SV(i,:) = result.SV;

            ds.CoJPoDs(i,:) = result.CoJPoDs;
        end
    else
        jpod_avg = mean(cellfun(@(x)x.JPoD,results,'UniformOutput',true),2,'omitnan');
        fsi_avg = mean(cellfun(@(x)x.FSI,results,'UniformOutput',true),2,'omitnan');
        pce_avg = mean(cellfun(@(x)x.PCE,results,'UniformOutput',true),2,'omitnan');
        ds.Indicators = [jpod_avg fsi_avg pce_avg];
        
        dide = cellfun(@(x)x.DiDe,results,'UniformOutput',false);
        si = cellfun(@(x)x.SI,results,'UniformOutput',false);
        sv = cellfun(@(x)x.SI,results,'UniformOutput',false);

        cojpods = cellfun(@(x)x.CoJPoDs,results,'UniformOutput',false);
        
        for i = 1:t
            ds.DiDe{i} = mean(cat(3,dide{i,:}),3);
            ds.SI(i,:) = mean(cat(3,si{i,:}),3);
            ds.SV(i,:) = mean(cat(3,sv{i,:}),3);

            ds.CoJPoDs(i,:) = mean(cat(3,cojpods{i,:}),3);
        end
    end
    
    ds.AverageDiDe = sum(cat(3,ds.DiDe{:}),3) ./ numel(ds.DiDe);

    si_vec = ds.SI(:);
    si_max = max(si_vec,[],'omitnan');
    si_min = min(si_vec,[],'omitnan');
    ds.SI = (ds.SI - si_min) ./ (si_max - si_min);

    sv_vec = ds.SV(:);
    sv_max = max(sv_vec,[],'omitnan');
    sv_min = min(sv_vec,[],'omitnan');
    ds.SV = (ds.SV - sv_min) ./ (sv_max - sv_min);

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

    labels_all = ds.PortfolioComponents;

    if (size(ds.Portfolios,1) == 1)
        if (strcmp(ds.SEL,'G'))
            header = {'Groups'};
        else
            header = {'Firms'};
        end
    else
        header = {'Components'};
    end

	labels = ds.LabelsIndicatorsSimple;
    tab = [dates_str array2table(ds.Indicators,'VariableNames',labels)];
    writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{1},'WriteRowNames',true);

    vars = [labels_all.' num2cell(ds.AverageDiDe)];
    tab = cell2table(vars,'VariableNames',[header labels_all]);
    writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{2},'WriteRowNames',true);

    tab = [dates_str array2table(ds.SI,'VariableNames',labels_all)];
    writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{3},'WriteRowNames',true);

    tab = [dates_str array2table(ds.SV,'VariableNames',labels_all)];
    writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{4},'WriteRowNames',true);
    
    tab = [dates_str array2table(ds.CoJPoDs,'VariableNames',labels_all)];
    writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{5},'WriteRowNames',true);
    
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

%% PLOTTING

function plot_portfolios_coverage(ds,id)

    if (strcmp(ds.SEL,'G'))
        n = ds.Groups;
        labels = ds.GroupShortNames.';
    else
        n = ds.N;
        labels = ds.FirmNames;
    end
    
    nc = numel(ds.PortfolioComponents);

    k = size(ds.Portfolios,1);
    
    seq = 1:n;

    counts = cell(k + 1,2);
    count_total = zeros(n,1);

    for i = 1:k
        pf_indices = ds.Portfolios{i,4};
        pf_indices = pf_indices(:);
        
        [values_unique,~,indices_unique] = unique(pf_indices);

        count = zeros(n,1);
        count(values_unique) = accumarray(indices_unique,1);
        
        count_total = count_total + count;
        
        [count,order] = sort(count);
        counts(i + 1,:) = {(count ./ (sum(count) / nc)) labels(order)};
    end

    [count_total,order] = sort(count_total);
    counts(1,:) = {(count_total ./ (sum(count_total) / nc)) labels(order)};
    
    if (k == 4)
        spp = [2 3];
        spo = {[1 4] 2 3 5 6};
    elseif (k == 3)
        spp = [3 2];
        spo = {[1 5] 2 4 6};
    else
        spp = [2 2];
        spo = {[1 3] 2 4};
    end

    f = figure('Name','Cross-Entropy Measures > Reduced Portfolios Coverage','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    subs = gobjects(k + 1,1);
    
    sub_1 = subplot(spp(1),spp(2),spo{1});
    bar(sub_1,seq,counts{1,1},'FaceColor',[0.749 0.862 0.933]);
    set(sub_1,'XLim',[0 (ds.N + 1)],'XTick',seq,'XTickLabel',counts{1,2},'XTickLabelRotation',90);
    set(sub_1,'YLim',[0 1],'YTick',0:0.1:1,'YTickLabels',arrayfun(@(x)sprintf('%.f%%',x),(0:0.1:1) .* 100,'UniformOutput',false));
    title('Overall');
    
    subs(1) = sub_1;
    
    for i = 1:k
        sub = subplot(spp(1),spp(2),spo{i + 1});
        bar(sub,seq,counts{i + 1,1},'FaceColor',[0.749 0.862 0.933]);
        set(sub,'XLim',[0 (ds.N + 1)],'XTick',seq,'XTickLabel',counts{i + 1,2},'XTickLabelRotation',90);
        set(sub,'YLim',[0 1],'YTick',0:0.1:1,'YTickLabels',arrayfun(@(x)sprintf('%.f%%',x),(0:0.1:1) .* 100,'UniformOutput',false));
        title(ds.Portfolios{i,1});
        
        subs(i + 1) = sub;
    end

    set(subs,'YGrid','on');

    figure_title('Reduced Portfolios Coverage');

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_indicators(ds,id)

    nc = numel(ds.PortfolioComponents);

    jpod = smooth_data(ds.Indicators(:,1));
    fsi = smooth_data(ds.Indicators(:,2));
    pce = smooth_data(ds.Indicators(:,3));

    f = figure('Name','Cross-Entropy Measures > Indicators','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);
    
    sub_1 = subplot(2,2,1:2);
    plot(sub_1,ds.DatesNum,jpod);
    set(sub_1,'YLim',plot_limits(jpod,0.1,0));
    t1 = title(sub_1,ds.LabelsIndicatorsSimple{1});
    set(t1,'Units','normalized');
    t1_position = get(t1,'Position');
    set(t1,'Position',[0.4783 t1_position(2) t1_position(3)]);
    
    sub_2 = subplot(2,2,3);
    plot(sub_2,ds.DatesNum,fsi);
    set(sub_2,'YLim',[1 nc],'YTick',1:nc);
    title(sub_2,ds.LabelsIndicatorsSimple{2});
    
    sub_3 = subplot(2,2,4);
    plot(sub_3,ds.DatesNum,pce);
    set(sub_3,'YLim',[0 1],'YTick',0:0.1:1,'YTickLabels',arrayfun(@(x)sprintf('%.f%%',x),(0:0.1:1) .* 100,'UniformOutput',false));
    title(sub_3,ds.LabelsIndicatorsSimple{3});
    
    set([sub_1 sub_2 sub_3],'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
    set([sub_1 sub_2 sub_3],'XGrid','on','YGrid','on');
    
    if (ds.MonthlyTicks)
        date_ticks([sub_1 sub_2 sub_3],'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        date_ticks([sub_1 sub_2 sub_3],'x','yyyy','KeepLimits');
    end

    figure_title(['Indicators (RR=' num2str(ds.RR * 100) '%, ' ds.PW ', ' ds.MD ')']);

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_dide(ds,id)

    nc = numel(ds.PortfolioComponents);

    dide = ds.AverageDiDe;
    didev = dide(:);
    
    [dide_x,dide_y] = meshgrid(1:nc,1:nc);
    dide_x = dide_x(:) + 0.5;
    dide_y = dide_y(:) + 0.5;
    
    dide_txt = cellstr(num2str(didev,'~%.4f'));

    for i = 1:nc^2
        didev_i = didev(i);

        if (didev_i == 0)
            dide_txt{i} = '0';
        elseif (didev_i == 1)
            dide_txt{i} = '';
        end
    end
    
    lt_indices = (dide < 0.2) & (dide ~= 1);
    ge_indices = (dide >= 0.2) & (dide ~= 1);
    
    dide(lt_indices) =  0;
    dide(ge_indices) =  1;
    dide = dide - (eye(nc) .* 0.5);
    dide = padarray(dide,[1 1],'post');
    
    didev = dide(:);
    didev_ones = any(didev == 1);
    didev_zeros = any(didev == 0);

    f = figure('Name','Cross-Entropy Measures > Average Distress Dependency','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    pcolor(dide);
    
    if (didev_ones && didev_zeros)
        colormap([1 1 1; 0.65 0.65 0.65; 0.749 0.862 0.933]);
    else
        if (didev_ones)
            colormap([0.65 0.65 0.65; 0.749 0.862 0.933]);
        else
            colormap([1 1 1; 0.65 0.65 0.65]);
        end
    end
        
    text(dide_x,dide_y,dide_txt,'HorizontalAlignment','center');
    axis image;

    ax = gca();
    set(ax,'TickLength',[0 0]);
    set(ax,'XAxisLocation','top','XTick',1.5:(nc + 0.5),'XTickLabels',ds.PortfolioComponents,'XTickLabelRotation',45);
    set(ax,'YDir','reverse','YTick',1.5:(nc + 0.5),'YTickLabels',ds.PortfolioComponents,'YTickLabelRotation',45);
    
    figure_title('Average Distress Dependency');

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_sequence_dide(ds,id)

    nc = numel(ds.PortfolioComponents);
    
    t = ds.T;
    dn = ds.DatesNum;
    mt = ds.MonthlyTicks;
    
    ts_si = ds.SI;
    ts_sv = ds.SV;

    data = [repmat({dn},1,nc); mat2cell(ts_si,t,ones(1,nc)); mat2cell(ts_sv,t,ones(1,nc))];
    
    sequence_titles = ds.PortfolioComponents;
	
	plots_title = [repmat(ds.LabelsMeasures(1),1,nc); repmat(ds.LabelsMeasures(2),1,nc)];

    x_limits = [dn(1) dn(end)];

    y_limits = [0 1];
    y_ticks = 0:0.1:1;

    core = struct();

    core.N = nc;
    core.Data = data;
    core.Function = @(subs,data)plot_function(subs,data);

    core.OuterTitle = 'Cross-Entropy Measures > Distress Dependency Time Series';
    core.InnerTitle = 'Distress Dependency Time Series';
    core.SequenceTitles = sequence_titles;

    core.PlotsAllocation = [2 1];
    core.PlotsSpan = {1 2};
    core.PlotsTitle = plots_title;

    core.XDates = {mt mt};
    core.XGrid = {true true};
    core.XLabel = {[] []};
    core.XLimits = {x_limits x_limits};
    core.XRotation = {[] []};
    core.XTick = {[] []};
    core.XTickLabels = {[] []};

    core.YGrid = {true true};
    core.YLabel = {[] []};
    core.YLimits = {y_limits y_limits};
    core.YRotation = {[] []};
    core.YTick = {y_ticks y_ticks};
    core.YTickLabels = {[] []};

    sequential_plot(core,id);
    
    function plot_function(subs,data)
        
        x = data{1};
        si = data{2};
        sv = data{3};
        
        d = min(find(isnan(si),1,'first'),find(isnan(sv),1,'first'));
        
        if (isempty(d))
            xd = [];
        else
            xd = x(d) - 1;
        end

        plot(subs(1),x,si,'Color',[0.000 0.447 0.741]);

        if (~isempty(xd))
            hold(subs(1),'on');
                plot(subs(1),[xd xd],get(subs(1),'YLim'),'Color',[1 0.4 0.4]);
            hold(subs(1),'off');
        end
    
        plot(subs(2),x,sv,'Color',[0.000 0.447 0.741]);

        if (~isempty(xd))
            hold(subs(2),'on');
                plot(subs(2),[xd xd],get(subs(2),'YLim'),'Color',[1 0.4 0.4]);
            hold(subs(2),'off');
        end
        
    end

end

function plot_sequence_cojpods(ds,id)

    nc = numel(ds.PortfolioComponents);

    t = ds.T;
    dn = ds.DatesNum;
    mt = ds.MonthlyTicks;
    
    ts = smooth_data(ds.CoJPoDs);

    data = [repmat({dn},1,nc); mat2cell(ts,t,ones(1,nc))];
    
    sequence_titles = ds.PortfolioComponents;
    
    plots_title = repmat(ds.LabelsMeasures(3),1,nc);
    
    x_limits = [dn(1) dn(end)];
	y_limits = plot_limits(ts,0.1,0);
    
    y_tick_labels = @(x)sprintf('%.2f%%',x .* 100);

    core = struct();

    core.N = nc;
    core.Data = data;
    core.Function = @(subs,data)plot_function(subs,data);

    core.OuterTitle = 'Cross-Entropy Measures > CoJPoDs Time Series';
    core.InnerTitle = 'CoJPoDs Time Series';
    core.SequenceTitles = sequence_titles;

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
    core.YTickLabels = {y_tick_labels};

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

%% VALIDATION

function out = validate_output(out)

    [path,name,extension] = fileparts(out);

    if (~strcmp(extension,'.xlsx'))
        out = fullfile(path,[name extension '.xlsx']);
    end
    
end

function sel = validate_selection(sel,groups)

    if (strcmp(sel,'G') && (groups == 0))
        error('The selection cannot be set to groups because their are not defined in the dataset.');
    end

end

function temp = validate_template(temp)

    sheets = {'Indicators' 'Average DiDe' 'SI' 'SV' 'CoJPoDs'};
    file_sheets = validate_xls(temp,'T');

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
