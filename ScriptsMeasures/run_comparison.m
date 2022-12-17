% [INPUT]
% ds = A structure representing the dataset.
% sn = A string representing the serial number of the result file.
% temp = A string representing the full path to the Excel spreadsheet used as template for the result file.
% out = A string representing the full path to the Excel spreadsheet to which the results are written, eventually replacing the previous ones.
% ml = A cell array of strings representing the measures labels.
% md = A float t-by-n matrix containing the measures time series.
% co = An integer representing the comparison cut-off, a limit before which all the observations are discarded (optional, default=1).
% sc = Optional argument specified as a vector of floats [0,Inf) of length 4 representing the score coefficient of each comparison model (Granger-causality, Logistic, Predictive Power Score, Price Discovery).
%      When defined, comparison models with a coefficient equal to 0 are not computed. When left undefined, all the comparison models are computed and their scores are equally weighted.  
% lag_max = An integer [2,Inf) representing the maximum lag order to be evaluated for Granger-causality and Price Discovery models (optional, default=10).
% lag_sel = A string representing the lag order selection criteria for Granger-causality and Price Discovery models (optional, default='AIC'):
%   - 'AIC' for Akaike's Information Criterion.
%   - 'BIC' for Bayesian Information Criterion.
%   - 'FPE' for Final Prediction Error.
%   - 'HQIC' for Hannan-Quinn Information Criterion.
% gca = A float [0.01,0.10] representing the probability level of the F test critical value used in the Granger-causality model (optional, default=0.01).
% lma = A boolean that indicates whether to use the adjusted McFadden R2 for the Logistic model (optional, default=false).
% ppsk = An integer [2,10] representing the number of cross-validation folds used in the Predictive Power Score model (optional, default=4).
% pdt = A string representing the type of metric to calculate for the Price Discovery model (optional, default='GG'):
%   - 'GG' for Gonzalo-Granger Component Metric.
%   - 'H' for Hasbrouck Information Metric.
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
        ip.addRequired('sn',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
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
        ip.addOptional('ppsk',4,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 2 '<=' 10 'scalar'}));
        ip.addOptional('pdt','GG',@(x)any(validatestring(x,{'GG' 'H'})));
        ip.addOptional('analyze',false,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    ds = validate_dataset(ipr.ds,'Comparison');
    sn = ipr.sn;
    temp = validate_template(ipr.temp);
    out = validate_output(ipr.out);
    [ml,md,co] = validate_measures(ipr.ml,ipr.md,ipr.co);
    sc = validate_sc(ipr.sc);
    lag_max = validate_lag_max(ipr.lag_max,md);
    lag_sel = ipr.lag_sel;
    gca = ipr.gca;
    lma = ipr.lma;
    ppsk = ipr.ppsk;
    pdt = ipr.pdt;
    analyze = ipr.analyze;

    nargoutchk(1,2);

    [result,stopped] = run_comparison_internal(ds,sn,temp,out,ml,md,co,sc,lag_max,lag_sel,gca,lma,ppsk,pdt,analyze);

end

function [result,stopped] = run_comparison_internal(ds,sn,temp,out,ml,md,co,sc,lag_max,lag_sel,gca,lma,ppsk,pdt,analyze)

    result = [];
    stopped = false;
    e = [];

    ds = initialize(ds,sn,ml,md,co,sc,lag_max,lag_sel,gca,lma,ppsk,pdt);
    k = numel(ds.SM) + 1;

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

            if (i == 1)
                ds = check_similarity(ds);
            else
                eval(['ds = perform_comparison_' lower(ds.SM{i-1}) '(ds);']);
            end

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
        analyze_result(ds);
    end

    result = ds;

end

%% PROCESS

function ds = initialize(ds,sn,ml,md,co,sc,lag_max,lag_sel,gca,lma,ppsk,pdt)

    cd = ds.CrisesDummy;
    cd_empty = isempty(cd);

    mdn = ds.DatesNum(co:end);
    [mdy,~,~,~,~,~] = datevec(mdn);

    md = md(co:end,:);
    [mt,mn] = size(md);

    md = (md - repmat(mean(md,1),mt,1)) ./ repmat(std(md,1),mt,1);
    md(isnan(md)) = 0;

    c = nchoosek(1:mn,2);
    c_len = size(c,1);
    mc = cell(c_len,2);

    for i = 1:c_len
        c_i = c(i,:);
        mc(i,:) = {c_i md(:,c_i)};
    end

    ds.Result = 'Comparison';
    ds.ResultDate = now(); %#ok<TNOW1> 
    ds.ResultAnalysis = @(ds)analyze_result(ds);
    ds.ResultSerial = sn;

    ds.MN = mn;
    ds.MT = mt;
    ds.MData = md;
    ds.MDatesNum = mdn;
    ds.MLabels = ml;
    ds.MCombinations = mc;
    ds.MMonthlyTicks = numel(unique(mdy)) <= 3;

    if (cd_empty)
        ds.MDummy = [];
    else
        ds.MDummy = cd(co:end);
    end

    ds.CO = co;
    ds.LagMax = lag_max;
    ds.LagSel = lag_sel;

    ds.LabelsSheetsSimple = {'Distance Correlation' 'RMS Similarity' 'Scores'};
    ds.LabelsSheets = {'Distance Correlation' 'RMS Similarity' 'Scores'};

    ds.DistanceCorrelation = NaN(mn);
    ds.RMSSimilarity = NaN(mn);

    ds.SC = sc;
    ds.SM = {'GC' 'LM' 'PPS' 'PD'};

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

    if ((ds.SC(off) > 0) && ~cd_empty)
        ds.LMA = lma;
        ds.LMData = NaN(1,mn);
        ds.LMScores = NaN(1,mn);
        off = off + 1;
    else
        ds.SC(off) = [];
        ds.SM(off) = [];
    end

    if ((ds.SC(off) > 0) && ~cd_empty)
        ds.PPSK = ppsk;
        ds.PPSData = NaN(1,mn);
        ds.PPSScores = NaN(1,mn);
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

    labels = [{'Firms'} ds.MLabels.'];

    if (ispc())
        xlswrite(out,[labels; ds.MLabels num2cell(ds.DistanceCorrelation)],1); %#ok<XLSWT> 
        xlswrite(out,[labels; ds.MLabels num2cell(ds.RMSSimilarity)],2); %#ok<XLSWT> 
    else
        xlswrite(out,[labels; ds.MLabels num2cell(ds.DistanceCorrelation)],ds.LabelsSheetsSimple{1}); %#ok<XLSWT> 
        xlswrite(out,[labels; ds.MLabels num2cell(ds.RMSSimilarity)],ds.LabelsSheetsSimple{2}); %#ok<XLSWT> 
    end

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
    writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{3},'WriteRowNames',true);

    worksheets_batch(out,ds.LabelsSheetsSimple,ds.LabelsSheets);

end

%% INTERNAL CALCULATIONS

function ds = check_similarity(ds)

    mc = ds.MCombinations;
    mc_len = size(mc,1);
    mc_results = cell(mc_len,2);

    parfor k = 1:mc_len
        md_k = mc{k,2};
        [dcor_i,rmss_i] = similarity_statistics(md_k);
        mc_results(k,:) = {dcor_i rmss_i};
    end

    mn = ds.MN;
    dcor = zeros(mn);
    rmss = zeros(mn);

    for k = 1:mc_len
        mo_k = mc{k,1};
        i = mo_k(1);
        j = mo_k(2);

        [dcor(i,j),rmss(i,j)] = deal(mc_results{k,:});
    end

    ds.DistanceCorrelation = dcor + dcor.' + eye(mn);
    ds.RMSSimilarity = rmss + rmss.' + eye(mn);

end

function ds = perform_comparison_gc(ds)

    lag_max = ds.LagMax;
    lag_sel = ds.LagSel;
    gca = ds.GCA;

    mc = ds.MCombinations;
    mc_len = size(mc,1);
    mc_results = cell(mc_len,1);

    parfor k = 1:mc_len
        md_k = mc{k,2};

        [h0,stat,cv,lag_r,lag_u] = granger_causality(md_k(:,1),md_k(:,2),gca,lag_max,lag_sel);

        data_k = struct();
        data_k.H0 = h0;
        data_k.Stat = stat;
        data_k.CV = cv;
        data_k.LagR = lag_r;
        data_k.LagU = lag_u;

        mc_results{k} = data_k;
    end

    mn = ds.MN;
    data = cell(mn,mn);
    scores = zeros(1,mn);

    for k = 1:mc_len
        mc_k = mc{k,1};
        i = mc_k(1);
        j = mc_k(2);

        data_k = mc_results{k};
        h0_k = data_k.H0;
        data_k = rmfield(data_k,'H0');

        data{i,j} = data_k;
        data{j,i} = data_k;

        if (~h0_k)
            scores(i) = scores(i) - 1;
            scores(j) = scores(j) + 1;
        end
    end

    if (~all(scores == 0))
        scores = round(((scores - min(scores)) ./ (max(scores) - min(scores))) .* 100,2);
    end

    ds.GCData = data;
    ds.GCScores = scores;

end

function ds = perform_comparison_lm(ds)

    m = ds.MData;
    [mt,mn] = size(m);

    lma = ds.LMA;
    y = ds.MDummy;

    mfr2 = zeros(1,mn);

    parfor i = 1:mn
        mdl = fitglm(m(:,i),y,'Distribution','binomial','Link','logit');
        b = mdl.Coefficients{:,1};
        ll = mdl.LogLikelihood;

        mdl0 = fitglm(zeros(mt,1),y,'Distribution','binomial','Link','logit');
        ll0 = mdl0.LogLikelihood;

        if (lma)
            mfr2(i) = 1 - ((ll - numel(b)) / ll0);
        else
            mfr2(i) = 1 - (ll / ll0);
        end
    end

    scores = zeros(1,mn);

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

function ds = perform_comparison_pps(ds)

    m = ds.MData;
    mn = size(m,2);

    ppsk = ds.PPSK;
    y = ds.MDummy;

    pps = zeros(1,mn);
    scores = zeros(1,mn);

    parfor i = 1:mn
        x = m(:,i);
        ts = [x y];

        if (size(ts,1) > 5000)
            ts = datasample(ts,5000,'Replace',false);
        end

        ts = ts(randperm(size(ts,1)),:);

        weight = 1 / size(ts,1);
        feature = ts(:,1);
        target = ts(:,2);

        baseline_loss = sum(weight .* (target - median(target)).^2);

        cvm = fitrtree(feature,target,'CrossVal','on','KFold',ppsk);
        loss = abs(kfoldLoss(cvm,'Mode','average','LossFun','mse'));

        if (loss > baseline_loss)
            pps(i) = 0;
        else
            pps(i) = 1 - (loss / baseline_loss);
        end
    end

    for i = 1:mn
        a = pps(i);

        for j = 1:mn
            if (i == j)
                continue;
            end

            b = pps(j);

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

    ds.PPSData = pps;
    ds.PPSScores = scores;

end

function ds = perform_comparison_pd(ds)

    lag_max = ds.LagMax;
    lag_sel = ds.LagSel;
    pdt = ds.PDT;

    mc = ds.MCombinations;
    mc_len = size(mc,1);
    mc_results = cell(mc_len,1);

    parfor k = 1:mc_len
        md_k = mc{k,2};

        [m1,m2,lag] = price_discovery(md_k,pdt,lag_max,lag_sel);

        data_k = struct();
        data_k.M1 = m1;
        data_k.M2 = m2;
        data_k.Lag = lag;

        mc_results{k} = data_k;
    end

    mn = ds.MN;
    data = cell(mn,mn);
    scores = zeros(1,mn);

    for k = 1:mc_len
        mo_k = mc{k,1};
        i = mo_k(1);
        j = mo_k(2);

        data_k = mc_results{k};
        data{i,j} = data_k;
        data{j,i} = data_k;

        if (data_k.M1 > 0.5)
            scores(i) = scores(i) + 1;
            scores(j) = scores(j) - 1;
        elseif (data_k.M2 > 0.5)
            scores(i) = scores(i) - 1;
            scores(j) = scores(j) + 1;
        end
    end

    if (~all(scores == 0))
        scores = round(((scores - min(scores)) ./ (max(scores) - min(scores))) .* 100,2);
    end

    ds.PDData = data;
    ds.PDScores = scores;

end

%% PLOTTING

function analyze_result(ds)

    k = numel(ds.SM);

    safe_plot(@(id)plot_measures(ds,id));
    safe_plot(@(id)plot_similarity(ds,'Distance Correlation',id));
    safe_plot(@(id)plot_similarity(ds,'RMS Similarity',id));

    for i = 1:k
        eval(['safe_plot(@(id)plot_scores_' lower(ds.SM{i}) '(ds,id));']);
    end

    if (k > 1)
        safe_plot(@(id)plot_scores_overall(ds,id));
    end

end

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

function plot_similarity(ds,target,id)

    labels = ds.MLabels;
    n = numel(labels);
    seq = 1:n;
    off = seq + 0.5;

    m = ds.(strrep(target,' ',''));

    s = m;
    s(s <= 0.5) = 0;
    s(s > 0.5) = 1;
    s(logical(eye(n))) = 0.5;

    [s_x,s_y] = meshgrid(seq,seq);
    s_x = s_x(:) + 0.5;
    s_y = s_y(:) + 0.5;
    s_text = cellstr(num2str(m(:),'%.2f'));

    f = figure('Name',['Measures Comparison > ' target],'Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    pcolor(padarray(s,[1 1],'post'));
    colormap([1 1 1; 0.65 0.65 0.65; 0.749 0.862 0.933]);

    if (n <= 10)
        axis('image');
    end

    text(s_x,s_y,s_text,'FontSize',9,'HorizontalAlignment','center');

    ax = gca();
    set(ax,'FontWeight','bold','TickLength',[0 0]);
    set(ax,'XAxisLocation','bottom','XTick',off,'XTickLabels',labels,'XTickLabelRotation',45);
    set(ax,'YDir','reverse','YTick',off,'YTickLabels',labels,'YTickLabelRotation',45);

    figure_title(target);

    pause(0.01);
    frame = get(f,'JavaFrame'); %#ok<JAVFM> 
    set(frame,'Maximized',true);

end

function plot_scores_gc(ds,id)

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
    frame = get(f,'JavaFrame'); %#ok<JAVFM> 
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
            tooltip = sprintf('%s\nF: %.4f | CV: %.4f\nRestricted Lag: %d\nUnrestricted Lag: %d',vs,data_ij.Stat,data_ij.CV,data_ij.LagR,data_ij.LagU);
        end

    end

end

function plot_scores_lm(ds,id)

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
    frame = get(f,'JavaFrame'); %#ok<JAVFM> 
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

function plot_scores_pps(ds,id)

    mn = ds.MN;
    seq = 1:mn;

    [scores,order] = sort(ds.PPSScores);
    labels = ds.MLabels(order);
    pps = ds.PPSData(order);

    f = figure('Name','Measures Comparison > Predictive Power Score Model','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    b = bar(seq,scores,'FaceColor',[0.749 0.862 0.933]);
    t = title('Scores');
    set(t,'Units','normalized');
    t_position = get(t,'Position');
    set(t,'Position',[0.4783 t_position(2) t_position(3)]);

    ax = gca();
    set(ax,'XLim',[0 (mn + 1)],'XTick',seq,'XTickLabel',labels,'XTickLabelRotation',45);
    set(ax,'YGrid','on','YLim',[0 100]);

    figure_title(['Predictive Power Score Model (K=' num2str(ds.PPSK) ')']);

    pause(0.01);
    frame = get(f,'JavaFrame'); %#ok<JAVFM> 
    set(frame,'Maximized',true);

    drawnow();

    dcm = datacursormode(f);
    set(dcm,'Enable','on','UpdateFcn',@(targ,evtd)create_tooltip(targ,evtd,pps));
    createDatatip(dcm,b,[1 1]);

    function tooltip = create_tooltip(~,evtd,mfr2)

        index = get(evtd,'DataIndex');
        tooltip = sprintf('PPS: %.4f',mfr2(index));

    end

end

function plot_scores_pd(ds,id)

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
    frame = get(f,'JavaFrame'); %#ok<JAVFM> 
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
        eval(['sm_scores(i,:) = ds.' ds.SM{i} 'Scores;']); %#ok<EVLDOT> 
    end

    sm_scores = sm_scores(:,order);

    ctooltips = cell(mn,1);

    for i = 1:mn
        ctooltip = cell(sm_len,1);

        for j = 1:sm_len
            ctooltip{j} = [ds.SM{j} ': ' sprintf('%.2f',sm_scores(j,i))];
        end

        ctooltips{i} = strjoin(ctooltip,new_line());
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
    frame = get(f,'JavaFrame'); %#ok<JAVFM> 
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

%% VALIDATION

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

    if (~strcmpi(extension,'.xlsx'))
        out = fullfile(path,[name extension '.xlsx']);
    end

end

function sc = validate_sc(sc)

    if (isempty(sc))
        sc = ones(1,4);
    else
        if (numel(sc) ~= 4)
            error('The ''sc'' parameter, when specified as a vector of floats, must contain exactly 4 elements.');
        end

        sc_sum = sum(sc);

        if (sc_sum == 0)
            error('The ''sc'' parameter, when specified as a vector of floats, must contain at least one positive element.');
        end
    end

end

function temp = validate_template(temp)

    sheets = {'Distance Correlation' 'RMS Similarity' 'Scores'};
    file_sheets = validate_xls(temp,'T');

    if (~all(ismember(sheets,file_sheets)))
        error(['The template must contain the following sheets: ' sheets{1} sprintf(', %s',sheets{2:end}) '.']);
    end

    worksheets_batch(temp,sheets);

end
