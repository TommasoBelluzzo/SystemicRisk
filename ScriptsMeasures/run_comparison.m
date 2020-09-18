% [INPUT]
% ds = A structure representing the dataset.
% temp = A string representing the full path to the Excel spreadsheet used as a template for the results file.
% out = A string representing the full path to the Excel spreadsheet to which the results are written, eventually replacing the previous ones.
% ml = A cell array of strings representing the measures labels.
% md = A float t-by-n matrix containing the measures time series.
% co = An integer representing the comparison cut-off, a limit before which all the observations are discarded (optional, default=1).
% sc = Optional argument specified as a vector of floats [0,Inf) of length 3 representing the score coefficient of each comparison model (Granger-causality, Logistic, Price Discovery).
%      When defined, comparison models with a coefficient equal to 0 are not computed. When left undefined, all the comparison models are computed and their scores are equally weighted.  
% lag_max = An integer [2,Inf) representing the maximum lag order to be evaluated for Granger-causality and Price Discovery models (optional, default=10).
% lag_sel = A string ('AIC', 'BIC', 'FPE' or 'HQIC') representing the lag order selection criteria for Granger-causality and Price Discovery models (optional, default='AIC').
% gca = A float [0.01,0.10] representing the probability level of the F test critical value for the Granger-causality model (optional, default=0.01).
% lma = A boolean that indicates whether to use the adjusted McFadden R2 for the Logistic model (optional, default=false).
% pdt = A string (either 'GG' for the Gonzalo-Granger component share or 'H' for the Hasbrouck information share) representing the type of metric to calculate for the Price Discovery model (optional, default='GG').
% analyze = A boolean that indicates whether to analyse the results and display plots (optional, default=false).
%
% [OUTPUT]
% result = A structure representing the original dataset inclusive of intermediate and final calculations.
% stopped = A boolean that indicates whether the process has been stopped through user input.

function [result,stopped] = run_comparison(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('ds',@(x)validateattributes(x,{'struct'},{'nonempty'}));
        ip.addRequired('temp',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addRequired('out',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addRequired('ml',@(x)validateattributes(x,{'cell'},{'vector' 'nonempty'}));
        ip.addRequired('md',@(x)validateattributes(x,{'double'},{'real' '2d' 'nonempty'}));
        ip.addOptional('co',1,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 1 'scalar'}));
        ip.addOptional('sc',[],@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0}));
        ip.addOptional('lag_max',10,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 1 'scalar'}));
        ip.addOptional('lag_sel','AIC',@(x)any(validatestring(x,{'AIC' 'BIC' 'FPE' 'HQIC'})));
        ip.addOptional('gca',0.01,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.01 '<=' 0.10 'scalar'}));
        ip.addOptional('lma',false,@(x)validateattributes(x,{'logical'},{'scalar'}));
        ip.addOptional('pdt','GG',@(x)any(validatestring(x,{'GG' 'H'})));
        ip.addOptional('analyze',false,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    ds = validate_dataset(ipr.ds);
    temp = validate_template(ipr.temp);
    out = validate_output(ipr.out);
    [ml,md,co] = validate_measures(ipr.ml,ipr.md,ipr.co);
    sc = validate_sc(ipr.sc);
    lag_max = validate_lag_max(ipr.lag_max,md);
    lag_sel = ipr.lag_sel;
    gca = ipr.gca;
    lma = ipr.lma;
    pdt = ipr.pdt;
    analyze = ipr.analyze;
    
    nargoutchk(1,2);

    [result,stopped] = run_comparison_internal(ds,temp,out,ml,md,co,sc,lag_max,lag_sel,gca,lma,pdt,analyze);

end

function [result,stopped] = run_comparison_internal(ds,temp,out,ml,md,co,sc,lag_max,lag_sel,gca,lma,pdt,analyze)

    result = [];
    stopped = false;
    e = [];

    ds = initialize(ds,ml,md,co,sc,lag_max,lag_sel,gca,lma,pdt);
    k = numel(ds.SM);

    bar = waitbar(0,'Initializing measures comparison...','CreateCancelBtn',@(src,event)setappdata(gcbf(),'Stop', true));
    setappdata(bar,'Stop',false);
    cleanup = onCleanup(@()delete(bar));
    
    pause(1);
    waitbar(0,bar,['Performing measures comparison (step 1 of ' num2str(k) ')...']);
    pause(1);

    try

        for i = 1:k
            waitbar((i - 1) / k,bar,['Performing measures comparison (step ' num2str(i) ' of ' num2str(k) ')...']);
            
            if (getappdata(bar,'Stop'))
                stopped = true;
                break;
            end
            
            eval(['ds = perform_comparison_' lower(ds.SM{i}) '(ds);']);
            
            if (getappdata(bar,'Stop'))
                stopped = true;
                break;
            end
            
            waitbar(i / k,bar);
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
    waitbar(1,bar,'Finalizing measures comparison...');
    pause(1);

    try
        ds = finalize(ds);
    catch e
        delete(bar);
        rethrow(e);
    end
    
    pause(1);
    waitbar(1,bar,'Writing measures comparison result...');
    pause(1);
    
    try
        write_results(ds,temp,out);
        delete(bar);
    catch e
        delete(bar);
        rethrow(e);
    end
    
    if (analyze)
        safe_plot(@(id)plot_measures(ds,id));
        
        for i = 1:k
            eval(['safe_plot(@(id)plot_scores_' lower(ds.SM{i}) '(ds,id));']);
        end
        
        if (k > 1)
            safe_plot(@(id)plot_scores_overall(ds,id));
        end
    end
    
    result = ds;

end

%% DATA

function ds = initialize(ds,ml,md,co,sc,lag_max,lag_sel,gca,lma,pdt)

    cd = ds.CrisesDummy;
    
    mdn = ds.DatesNum(co:end);
    [mdy,~,~,~,~,~] = datevec(mdn);

    md = md(co:end,:);
    [mt,mn] = size(md);

    md = (md - repmat(mean(md,1),mt,1)) ./ repmat(std(md,1),mt,1);
    md(isnan(md)) = 0;

    ds.MN = mn;
    ds.MT = mt;
    ds.MData = md;
    ds.MDatesNum = mdn;
    ds.MLabels = ml;
    ds.MMonthlyTicks = numel(unique(mdy)) <= 3;
    
    if (~isempty(cd))
        ds.MDummy = cd(co:end);
    end

    ds.SC = sc;
    ds.SM = {'GC' 'LM' 'PD'};

    ds.CO = co;
    ds.LagMax = lag_max;
    ds.LagSel = lag_sel;

    off = 1;

    if (ds.SC(off) > 0)
        ds.GCA = gca;
        ds.GCData = cell(mn,mn);
        ds.GCScores = NaN(1,mn);
        off = off + 1;
    else
        ds.SC(off) = [];
        ds.SM(off) = [];
    end
    
    if ((ds.SC(off) > 0) && ~isempty(ds.MDummy))
        ds.LMA = lma;
        ds.LMData = NaN(1,mn);
        ds.LMScores = NaN(1,mn);
        off = off + 1;
    else
        ds.SC(off) = [];
        ds.SM(off) = [];
    end

    if (ds.SC(off) > 0)
        ds.PDT = pdt;
        ds.PDData = cell(mn,mn);
        ds.PDScores = NaN(1,mn);
    else
        ds.SC(off) = [];
        ds.SM(off) = [];
    end
    
    if (numel(ds.SM) > 1)
        ds.OverallScores = NaN(1,mn);
    end

end

function ds = finalize(ds)

    if (numel(ds.SM) > 1)
        mn = ds.MN;
        k = numel(ds.SM);

        overall_scores = zeros(1,mn);

        for i = 1:k
            overall_scores = overall_scores + (ds.([ds.SM{i} 'Scores']) .* ds.SC(i));
        end

        ds.OverallScores = round(overall_scores ./ sum(ds.SC),2);
    end

end

function lag_max = validate_lag_max(lag_max,md)

    mt = size(md,1);
    b = (lag_max * 2) + 1;
    
    if ((lag_max > (mt - 2)) || (b >= (mt - b)))
        error('The ''lag_max'' parameter is too high for the provided number of observations.');
    end

end

function [ml,md,co] = validate_measures(ml,md,co)

    if (any(cellfun(@(x)~ischar(x)||isempty(x),ml)))
        error('The ''ml'' parameter contains invalid values.');
    end

    [mt,mn] = size(md);

    if (mn ~= numel(ml))
        error('The number of measures and the number of measure labels are mismatching.');
    end

    if ((mt - co + 1) < 100)
        error('The number of observations, after the cut-off is applied, is too low to obtain consistent results.');
    end
    
    ml = ml(:);
    
end

function out = validate_output(out)

    [path,name,extension] = fileparts(out);

    if (~strcmp(extension,'.xlsx'))
        out = fullfile(path,[name extension '.xlsx']);
    end
    
end

function sc = validate_sc(sc)

    if (isempty(sc))
        sc = ones(1,3);
    else
        if (numel(sc) ~= 3)
            error('The ''sc'' parameter, when specified as a vector of floats, must contain exactly 3 values.');
        end
        
        sc_sum = sum(sc);
        
        if (sc_sum == 0)
            error('The ''sc'' parameter, when specified as a vector of floats, must contain at least one positive value.');
        end
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

    if ((numel(file_sheets) ~= 1) || ~strcmp(file_sheets{1},'Scores'))
        error('The template must contain only one sheet named ''Scores''.');
    end
    
    if (ispc())
        try
            excel = actxserver('Excel.Application');
            excel_wb = excel.Workbooks.Open(temp,0,false);
                
            excel_wb.Sheets.Item('Scores').Cells.Clear();
            
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
    
    mn = ds.MN;
    
    sm = ds.SM;
    sm_len = numel(sm);
    
    labels = [{'Measure'} sm];
    
    if (sm_len == 1)
        scores = ds.([sm{1} 'Scores']).';
    else
        labels = [labels {'Overall'}];
        
        scores = zeros(mn,sm_len + 1);
        
        for i = 1:sm_len
            scores(:,i) = ds.([sm{i} 'Scores']).';
        end
        
        scores(:,end) = ds.OverallScores.';
    end
    
    scores = mat2cell(scores,ds.MN,ones(1,size(scores,2)));
    
    tab = table(ds.MLabels,scores{:},'VariableNames',labels);
    writetable(tab,out,'FileType','spreadsheet','Sheet','Scores','WriteRowNames',true);

end

%% MEASURES

function ds = perform_comparison_gc(ds) %#ok<DEFNU>

    m = ds.MData;
    mn = size(m,2);
    
    lag_max = ds.LagMax;
    lag_sel = ds.LagSel;
    gca = ds.GCA;

    data = cell(mn,mn);
    scores = zeros(1,mn);

    for i = 1:mn
        m_i = m(:,i);

        for j = 1:mn
            if (i == j)
                continue;
            end

            m_j = m(:,j);

            [f,cv,h0,lag_r,lag_u] = granger_causality([m_i m_j],gca,lag_max,lag_sel);
            
            data_ij = struct();
            data_ij.F = f;
            data_ij.CV = cv;
            data_ij.LagR = lag_r;
            data_ij.LagU = lag_u;
            
            data{i,j} = data_ij;

            if (~h0)
                scores(i) = scores(i) - 1;
                scores(j) = scores(j) + 1;
            end
        end
    end
    
    if (~all(scores == 0))
        scores = round(((scores - min(scores)) ./ (max(scores) - min(scores))) .* 100,2);
    end
    
    ds.GCData = data;
    ds.GCScores = scores;

end

function ds = perform_comparison_lm(ds) %#ok<DEFNU>

    m = ds.MData;
    [mt,mn] = size(m);
    
    lma = ds.LMA;
    y = ds.MDummy;
    a = zeros(mt,1);

	mfr2 = zeros(1,mn);
    scores = zeros(1,mn);

    parfor i = 1:mn
        mdl = fitglm(m(:,i),y,'Distribution','binomial','Link','logit');
        b = mdl.Coefficients{:,1};
        ll = mdl.LogLikelihood;

        mdl0 = fitglm(a,y,'Distribution','binomial','Link','logit');
        ll0 = mdl0.LogLikelihood;

        if (lma)
            mfr2(i) = 1 - ((ll - numel(b)) / ll0);
        else
            mfr2(i) = 1 - (ll / ll0);
        end
    end

    for i = 1:mn
        a = mfr2(i);

        for j = 1:mn
            if (i == j)
                continue;
            end

            b = mfr2(j);

            tol = 1e4 * eps(min(abs([a b])));

            if (a > (b + tol))
                scores(i) = scores(i) + 1;
                scores(j) = scores(j) - 1;
            elseif (b > (a + tol))
                scores(i) = scores(i) - 1;
                scores(j) = scores(j) + 1;
            end
        end
    end
    
    if (~all(scores == 0))
        scores = round(((scores - min(scores)) ./ (max(scores) - min(scores))) .* 100,2);
    end

    ds.LMData = mfr2;
    ds.LMScores = scores;

end

function ds = perform_comparison_pd(ds) %#ok<DEFNU>

    m = ds.MData;
    mn = size(m,2);
    
    lag_max = ds.LagMax;
    lag_sel = ds.LagSel;
    pdt = ds.PDT;

    data = cell(mn,mn);
    scores = zeros(1,mn);

    for i = 1:mn
        m_i = m(:,i);

        for j = 1:mn
            if (i == j)
                continue;
            end

            m_j = m(:,j);

            [m1,m2,lag] = price_discovery([m_i m_j],pdt,lag_max,lag_sel);
            
            data_ij = struct();
            data_ij.M1 = m1;
            data_ij.M2 = m2;
            data_ij.Lag = lag;
            
            data{i,j} = data_ij;

            if (m1 > 0.5)
                scores(i) = scores(i) + 1;
                scores(j) = scores(j) - 1;
            elseif (m2 > 0.5)
                scores(i) = scores(i) - 1;
                scores(j) = scores(j) + 1;
            end
        end
    end
    
    if (~all(scores == 0))
        scores = round(((scores - min(scores)) ./ (max(scores) - min(scores))) .* 100,2);
    end
    
    ds.PDData = data;
    ds.PDScores = scores;

end

%% PLOTTING

function plot_measures(ds,id)

    ct = ds.CrisesType;

    mn = ds.MN;
    mt = ds.MT;
    mdn = ds.MDatesNum;
    mmt = ds.MMonthlyTicks;

    ts = smooth_data(ds.MData);
    ts_limits = plot_limits(ts,0.1);

    if (isfield(ds,'MDummy'))
        if (strcmp(ct,'E'))
            c1 = ds.MDatesNum(logical(ds.MDummy));
            c2 = ts_limits;
        else
            c1 = ds.MDummy .* ts_limits(2);
            c1(c1 == 0) = NaN;
            c2 = ts_limits(1);
        end
    else
        ct = '';
        c1 = [];
        c2 = [];
    end
    
    data = [repmat({mdn},1,mn); mat2cell(ts,mt,ones(1,mn))];
    
    sequence_titles = ds.MLabels.';

    plots_title = repmat({' '},1,mn);
    
    x_limits = [mdn(1) mdn(end)];
    y_limits = ts_limits;

    core = struct();

    core.N = mn;
    core.Data = data;
    core.Function = @(subs,data)plot_function(subs,data,ct,c1,c2);

    core.OuterTitle = 'Measures Comparison > Measures';
    core.InnerTitle = 'Measures Time Series';
    core.SequenceTitles = sequence_titles;

    core.PlotsAllocation = [1 1];
    core.PlotsSpan = {1};
    core.PlotsTitle = plots_title;

    core.XDates = {mmt};
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
    
    function plot_function(subs,data,ct,c1,c2)

        x = data{1};
        y = data{2};
        
        if (isempty(ct))
            plot(subs(1),x,y,'Color',[0.000 0.447 0.741]);
        else
            if (strcmp(ct,'E'))
                plot(subs(1),x,y,'Color',[0.000 0.447 0.741]);
                
                hold(subs(1),'on');
                    for i = 1:numel(c1)
                        line(subs(1),ones(2,1) .* c1(i),[c2(1) c2(2)],'Color',[1 0.4 0.4]);
                    end
                hold(subs(1),'off');
            else
                area(subs(1),x,c1,c2,'EdgeColor','none','FaceAlpha',0.4,'FaceColor',[0.850 0.325 0.098]);

                hold(subs(1),'on');
                    plot(subs(1),x,y,'Color',[0.000 0.447 0.741]);
                hold(subs(1),'off');
            end
        end

    end

end

function plot_scores_gc(ds,id) %#ok<DEFNU>

    mn = ds.MN;
    seq = 1:mn;
    off = seq + 0.5;
    
    [seq_x,seq_y] = meshgrid(seq,seq);
    seq_x = seq_x(:) + 0.5;
    seq_y = seq_y(:) + 0.5;
    seq_diag = seq_x == seq_y;

    [scores,order] = sort(ds.GCScores);
    labels = ds.MLabels(order);

    data_labels = ds.MLabels;
    data = ds.GCData;

    f = figure('Name','Measures Comparison > Granger-causality','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    sub_1 = subplot(1,2,1);
    bar(sub_1,seq,scores,'FaceColor',[0.749 0.862 0.933]);
    set(sub_1,'XLim',[0 (mn + 1)],'XTick',seq,'XTickLabel',labels,'XTickLabelRotation',90);
    set(sub_1,'YGrid','on','YLim',[0 100]);
    title('Scores');

    sub_2 = subplot(1,2,2);
    s = scatter(seq_x,seq_y,1000,[0.749 0.862 0.933],'.','MarkerFaceColor','none');
    hold on;
        cdata = s.CData;
        cdata = repmat(cdata(1,:),numel(seq_x),1);
        cdata(seq_diag,:) = repmat([0.65 0.65 0.65],sum(seq_diag),1);
        s.CData = cdata;
    hold off;
    set(sub_2,'TickLength',[0 0]);
    set(sub_2,'XLim',[0.5 (mn + 1.5)],'XTick',off,'XTickLabels',ds.MLabels,'XTickLabelRotation',90);
    set(sub_2,'YDir','reverse','YLim',[0.5 (mn + 1.5)],'YTick',off,'YTickLabels',ds.MLabels,'YTickLabelRotation',0);
    set(sub_2,'Box','on','XGrid','on','YGrid','on');
    title('Data Browser');
    
    figure_title(['Granger-causality (A=' num2str(ds.GCA * 100) '%, LM=' num2str(ds.LagMax) ', LS=' ds.LagSel ')']);

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

    drawnow();

    dcm = datacursormode(f);
    set(dcm,'Enable','on','SnapToDataVertex','off','UpdateFcn',@(targ,evtd)create_tooltip(targ,evtd,s,data_labels,data));
    createDatatip(dcm,s,[1 1]);

    function tooltip = create_tooltip(~,evtd,element,data_labels,data)

        targ = get(evtd,'Target');
        
        if (targ ~= element)
            tooltip = [];
            return;
        end
        
        [i,j] = ind2sub(size(data),get(evtd,'DataIndex'));
        data_ij = data{i,j};
        
        if (isempty(data_ij))
            tooltip = '';
        else
            vs = [data_labels{i} ' vs ' data_labels{j}];
            tooltip = sprintf('%s\nF: %.4f | CV: %.4f\nRestricted Lag: %d\nUnrestricted Lag: %d',vs,data_ij.F,data_ij.CV,data_ij.LagR,data_ij.LagU);
        end

    end

end

function plot_scores_lm(ds,id) %#ok<DEFNU>

    mn = ds.MN;
    seq = 1:mn;

    [scores,order] = sort(ds.LMScores);
    labels = ds.MLabels(order);
    mfr2 = ds.LMData(order);
    
    if (ds.LMA)
        lma_label = 'Y';
    else
        lma_label = 'N';
    end

    f = figure('Name','Measures Comparison > Logistic Model','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    b = bar(seq,scores,'FaceColor',[0.749 0.862 0.933]);
    t = title('Scores');
    set(t,'Units','normalized');
    t_position = get(t,'Position');
    set(t,'Position',[0.4783 t_position(2) t_position(3)]);
    
    ax = gca();
    set(ax,'XLim',[0 (mn + 1)],'XTick',seq,'XTickLabel',labels,'XTickLabelRotation',45);
    set(ax,'YGrid','on','YLim',[0 100]);

    figure_title(['Logistic Model (ADJ=' lma_label ')']);

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

    drawnow();

    dcm = datacursormode(f);
    set(dcm,'Enable','on','UpdateFcn',@(targ,evtd)create_tooltip(targ,evtd,mfr2));
    createDatatip(dcm,b,[1 1]);

    function tooltip = create_tooltip(~,evtd,mfr2)

        index = get(evtd,'DataIndex');
        tooltip = sprintf('MFR2: %.4f',mfr2(index));

    end

end

function plot_scores_pd(ds,id) %#ok<DEFNU>

    mn = ds.MN;
    seq = 1:mn;
    off = seq + 0.5;
    
    [seq_x,seq_y] = meshgrid(seq,seq);
    seq_x = seq_x(:) + 0.5;
    seq_y = seq_y(:) + 0.5;
    seq_diag = seq_x == seq_y;

    [scores,order] = sort(ds.PDScores);
    labels = ds.MLabels(order);

    data_labels = ds.MLabels;
    data = ds.PDData;

    f = figure('Name','Measures Comparison > Price Discovery','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    sub_1 = subplot(1,2,1);
    bar(sub_1,seq,scores,'FaceColor',[0.749 0.862 0.933]);
    set(sub_1,'XLim',[0 (mn + 1)],'XTick',seq,'XTickLabel',labels,'XTickLabelRotation',90);
    set(sub_1,'YGrid','on','YLim',[0 100]);
    title('Scores');

    sub_2 = subplot(1,2,2);
    s = scatter(seq_x,seq_y,1000,[0.749 0.862 0.933],'.','MarkerFaceColor','none');
    hold on;
        cdata = s.CData;
        cdata = repmat(cdata(1,:),numel(seq_x),1);
        cdata(seq_diag,:) = repmat([0.65 0.65 0.65],sum(seq_diag),1);
        s.CData = cdata;
    hold off;
    set(sub_2,'TickLength',[0 0]);
    set(sub_2,'XLim',[0.5 (mn + 1.5)],'XTick',off,'XTickLabels',ds.MLabels,'XTickLabelRotation',90);
    set(sub_2,'YDir','reverse','YLim',[0.5 (mn + 1.5)],'YTick',off,'YTickLabels',ds.MLabels,'YTickLabelRotation',0);
    set(sub_2,'Box','on','XGrid','on','YGrid','on');
    title('Data Browser');
    
    figure_title(['Price Discovery (' ds.PDT ', LM=' num2str(ds.LagMax) ', LS=' ds.LagSel ')']);

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

    drawnow();

    dcm = datacursormode(f);
    set(dcm,'Enable','on','SnapToDataVertex','off','UpdateFcn',@(targ,evtd)create_tooltip(targ,evtd,s,data_labels,data));
    createDatatip(dcm,s,[1 1]);

    function tooltip = create_tooltip(~,evtd,element,data_labels,data)

        targ = get(evtd,'Target');
        
        if (targ ~= element)
            tooltip = [];
            return;
        end
        
        [i,j] = ind2sub(size(data),get(evtd,'DataIndex'));
        data_ij = data{i,j};
        
        if (isempty(data_ij))
            tooltip = '';
        else
            vs = [data_labels{i} ' vs ' data_labels{j}];
            tooltip = sprintf('%s\nM1: %f\nM2: %f\nLag: %d',vs,data_ij.M1,data_ij.M2,data_ij.Lag);
        end

    end

end

function plot_scores_overall(ds,id)

    mn = ds.MN;
    seq = 1:mn;

    [scores,order] = sort(ds.OverallScores);
    labels = ds.MLabels(order);

    sm_len = numel(ds.SM);
    sm_scores = zeros(sm_len,mn);

    ctitles = cell(sm_len,1);

    for i = 1:sm_len
        ctitles{i} = [ds.SM{i} '=' num2str(ds.SC(i))];
        eval(['sm_scores(i,:) = ds.' ds.SM{i} 'Scores;']);
    end

    sm_scores = sm_scores(:,order);
    
    ctooltips = cell(mn,1);
    
    for i = 1:mn
        ctooltip = cell(sm_len,1);
        
        for j = 1:sm_len
            ctooltip{j} = [ds.SM{j} ': ' sprintf('%.2f',sm_scores(j,i))];
        end
        
        ctooltips{i} = strjoin(ctooltip,'\n');
    end

    f = figure('Name','Measures Comparison > Overall Scores','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    b = bar(seq,scores,'FaceColor',[0.749 0.862 0.933]);
    t = title(['Score Coefficients: ' strjoin(ctitles,' | ')]);
    set(t,'Units','normalized');
    t_position = get(t,'Position');
    set(t,'Position',[0.4783 t_position(2) t_position(3)]);
    
    ax = gca();
    set(ax,'XLim',[0 (mn + 1)],'XTick',seq,'XTickLabel',labels,'XTickLabelRotation',45);
    set(ax,'YGrid','on','YLim',[0 100]);

    figure_title('Overall Scores');

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

    drawnow();

    dcm = datacursormode(f);
    set(dcm,'Enable','on','UpdateFcn',@(targ,evtd)create_tooltip(targ,evtd,ctooltips));
    createDatatip(dcm,b,[1 1]);

    function tooltip = create_tooltip(~,evtd,ctooltips)

        index = get(evtd,'DataIndex');
        tooltip = ctooltips{index};

    end

end
