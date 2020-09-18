% [INPUT]
% ds = A structure representing the dataset.
% temp = A string representing the full path to the Excel spreadsheet used as a template for the results file.
% out = A string representing the full path to the Excel spreadsheet to which the results are written, eventually replacing the previous ones.
% rs2 = A boolean that indicates whether to compute the two-states regime-switching model (optional, default=true).
% rs3 = A boolean that indicates whether to compute the three-states regime-switching model (optional, default=true).
% rs4 = A boolean that indicates whether to compute the four-states regime-switching model (optional, default=true).
% analyze = A boolean that indicates whether to analyse the results and display plots (optional, default=false).
%
% [OUTPUT]
% result = A structure representing the original dataset inclusive of intermediate and final calculations.
% stopped = A boolean that indicates whether the process has been stopped through user input.

function [result,stopped] = run_regime_switching(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('ds',@(x)validateattributes(x,{'struct'},{'nonempty'}));
        ip.addRequired('temp',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addRequired('out',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addOptional('rs2',true,@(x)validateattributes(x,{'logical'},{'scalar'}));
        ip.addOptional('rs3',true,@(x)validateattributes(x,{'logical'},{'scalar'}));
        ip.addOptional('rs4',true,@(x)validateattributes(x,{'logical'},{'scalar'}));
        ip.addOptional('analyze',false,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    ds = validate_dataset(ipr.ds,'regime-switching');
    temp = validate_template(ipr.temp);
    out = validate_output(ipr.out);
    [rs2,rs3,rs4] = validate_booleans(ipr.rs2,ipr.rs3,ipr.rs4);
    analyze = ipr.analyze;
    
    nargoutchk(1,2);

    [result,stopped] = run_regime_switching_internal(ds,temp,out,rs2,rs3,rs4,analyze);

end

function [result,stopped] = run_regime_switching_internal(ds,temp,out,rs2,rs3,rs4,analyze)

    result = [];
    stopped = false;
    e = [];

    ds = initialize(ds,rs2,rs3,rs4);
    n = ds.N;
    t = ds.T;

    bar = waitbar(0,'Initializing regime-switching measures...','CreateCancelBtn',@(src,event)setappdata(gcbf(),'Stop', true));
    setappdata(bar,'Stop',false);
    cleanup = onCleanup(@()delete(bar));
    
    pause(1);
    waitbar(0,bar,'Calculating regime-switching measures...');
    pause(1);

    try

        for i = 1:n
            
            index = 1;
            offset = min(ds.Defaults(i) - 1,t);
            waitbar((i - 1) / n,bar,['Calculating regime-switching measures for ' ds.FirmNames{i} '...']);

            if (getappdata(bar,'Stop'))
                stopped = true;
                break;
            end

            r = ds.Returns(1:offset,i);

            if (ds.RS2)
                [mu_params_2,s2_params_2,p_2,sprob_2,dur_2,cmu_2,cs2_2] = regime_switching_2(r);

                ds.MeanParams{index,i} = mu_params_2;
                ds.ConditionalMeans{index,i} = [cmu_2; NaN(t - offset,1)];
                
                ds.VarianceParams{index,i} = s2_params_2;
                ds.ConditionalVariances{index,i} = [cs2_2; NaN(t - offset,1)];
                
                ds.TransitionMatrices{index,i} = p_2;
                ds.SmoothedProbabilities{index,i} = [sprob_2; NaN(t - offset,2)];
                ds.Durations{index,i} = dur_2;

                index = index + 1;
            end
            
            if (ds.RS3)
                [mu_params_3,s2_params_3,p_3,sprob_3,dur_3,cmu_3,cs2_3] = regime_switching_3(r);
                
                ds.MeanParams{index,i} = mu_params_3;
                ds.ConditionalMeans{index,i} = [cmu_3; NaN(t - offset,1)];

                ds.VarianceParams{index,i} = s2_params_3;
                ds.ConditionalVariances{index,i} = [cs2_3; NaN(t - offset,1)];

                ds.TransitionMatrices{index,i} = p_3;
                ds.SmoothedProbabilities{index,i} = [sprob_3; NaN(t - offset,3)];
                ds.Durations{index,i} = dur_3;

                index = index + 1;
            end
            
            if (ds.RS4)
                [mu_params_4,s2_params_4,p_4,sprob_4,dur_4,cmu_4,cs2_4] = regime_switching_4(r);
                
                ds.MeanParams{index,i} = mu_params_4;
                ds.ConditionalMeans{index,i} = [cmu_4; NaN(t - offset,1)];

                ds.VarianceParams{index,i} = s2_params_4;
                ds.ConditionalVariances{index,i} = [cs2_4; NaN(t - offset,1)];
                
                ds.TransitionMatrices{index,i} = p_4;
                ds.SmoothedProbabilities{index,i} = [sprob_4; NaN(t - offset,4)];
                ds.Durations{index,i} = dur_4;
            end

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
    waitbar(1,bar,'Finalizing regime-switching measures...');
    pause(1);

    try
        ds = finalize(ds);
    catch e
        delete(bar);
        rethrow(e);
    end
    
    pause(1);
    waitbar(1,bar,'Writing regime-switching measures...');
    pause(1);
    
    try
        write_results(ds,temp,out);
        delete(bar);
    catch e
        delete(bar);
        rethrow(e);
    end
    
    if (analyze)
        if (ds.RS2)
            safe_plot(@(id)plot_indicators(ds,'RS2',id));
            safe_plot(@(id)plot_sequence(ds,'RS2',id));
        end
        
        if (ds.RS3)
            safe_plot(@(id)plot_indicators(ds,'RS3',id));
            safe_plot(@(id)plot_sequence(ds,'RS3',id));
        end
        
        if (ds.RS4)
            safe_plot(@(id)plot_indicators(ds,'RS4',id));
            safe_plot(@(id)plot_sequence(ds,'RS4',id));
        end
    end

    result = ds;

end

%% PROCESS

function ds = initialize(ds,rs2,rs3,rs4)

    n = ds.N;
    t = ds.T;

    rs = [rs2 rs3 rs4];
    rs_seq = 2:4;
    rs_seq = arrayfun(@(x)sprintf('RS%d-',x),rs_seq(rs),'UniformOutput',false);
    
    m = sum(rs);
    
    ds.RS2 = rs2;
    ds.RS3 = rs3;
    ds.RS4 = rs4;
    
    ds.LabelsSheetsSimple = {'Indicators' 'RS2 CM' 'RS2 CV' 'RS2 SP' 'RS3 CM' 'RS3 CV' 'RS3 SP' 'RS4 CM' 'RS4 CV' 'RS4 SP'};
    ds.LabelsSheets = ds.LabelsSheetsSimple;
    
    ds.MeanParams = cell(m,n);
    ds.ConditionalMeans = cell(m,n);
    
    ds.VarianceParams = cell(m,n);
    ds.ConditionalVariances = cell(m,n);

    ds.TransitionMatrices = cell(m,n);
    ds.SmoothedProbabilities = cell(m,n);
    ds.Durations = cell(m,n);

    ds.AverageProbabilities = NaN(t,m);
    ds.JointProbabilities = NaN(t,m);
    
    ds.ComparisonReferences = {
        'AverageProbabilities' [] strcat(rs_seq,{'AP'});
        'JointProbabilities' [] strcat(rs_seq,{'JP'})
    };

end

function ds = finalize(ds)

    n = ds.N;
    index = 1;
    
    if (ds.RS2)
        sprob = cell2mat(ds.SmoothedProbabilities(index,:));
        sprob_hv = sprob(:,1:2:n*2);

        ds.AverageProbabilities(:,index) = mean(sprob_hv,2,'omitnan');
        ds.JointProbabilities(:,index) = prod(sprob_hv,2,'omitnan');
        
        index = index + 1;
    end

    if (ds.RS3)
        sprob = cell2mat(ds.SmoothedProbabilities(index,:));
        sprob_hv = sprob(:,1:3:n*3);

        ds.AverageProbabilities(:,index) = mean(sprob_hv,2,'omitnan');
        ds.JointProbabilities(:,index) = prod(sprob_hv,2,'omitnan');
        
        index = index + 1;
    end

    if (ds.RS4)
        sprob = cell2mat(ds.SmoothedProbabilities(index,:));
        sprob_hv = sprob(:,1:4:n*4) + sprob(:,2:4:n*4);

        ds.AverageProbabilities(:,index) = mean(sprob_hv,2,'omitnan');
        ds.JointProbabilities(:,index) = prod(sprob_hv,2,'omitnan');
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

    labels_regimes = repmat({'RS2' 'RS3' 'RS4'},1,2);
    labels_types = repelem({' AP' ' JP'},1,3);
    labels_indicators = strcat(labels_regimes,labels_types);
    
    if (~ds.RS2)
        labels_indicators(strcmp(labels_regimes,'RS2')) = [];
    end
    
    if (~ds.RS3)
        labels_indicators(strcmp(labels_regimes,'RS3')) = [];
    end
    
    if (~ds.RS4)
        labels_indicators(strcmp(labels_regimes,'RS4')) = [];
    end

    tab = [dates_str array2table([ds.AverageProbabilities ds.JointProbabilities],'VariableNames',strrep(labels_indicators,' ','_'))];
    writetable(tab,out,'FileType','spreadsheet','Sheet','Indicators','WriteRowNames',true);

    if (ds.RS2)
        if (sum([ds.RS2 ds.RS3 ds.RS4]) == 1)
            offset = 1;
        else
            sizes = cellfun(@(x)unique(size(x)),ds.TransitionMatrices(:,1));
            offset = find(sizes == 2,1,'first');
        end
        
        data_cmu = cell2mat(ds.ConditionalVariances(offset,:));
        
        tab = [dates_str array2table(data_cmu,'VariableNames',ds.FirmNames)];
        writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{2},'WriteRowNames',true);
        
        data_cs2 = cell2mat(ds.ConditionalVariances(offset,:));
        
        tab = [dates_str array2table(data_cs2,'VariableNames',ds.FirmNames)];
        writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{3},'WriteRowNames',true);

        data_sprob = cell2mat(ds.SmoothedProbabilities(offset,:));
        
        tab = [dates_str array2table(data_sprob,'VariableNames',strcat(repelem(ds.FirmNames,1,2),repmat({'_HV' '_LV'},1,ds.N)))];
        writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{4},'WriteRowNames',true);
    end

    if (ds.RS3)
        if (sum([ds.RS2 ds.RS3 ds.RS4]) == 1)
            offset = 1;
        else
            sizes = cellfun(@(x)unique(size(x)),ds.TransitionMatrices(:,1));
            offset = find(sizes == 3,1,'first');
        end
        
        data_cmu = cell2mat(ds.ConditionalVariances(offset,:));

        tab = [dates_str array2table(data_cmu,'VariableNames',ds.FirmNames)];
        writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{5},'WriteRowNames',true);

        data_cs2 = cell2mat(ds.ConditionalVariances(offset,:));

        tab = [dates_str array2table(data_cs2,'VariableNames',ds.FirmNames)];
        writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{6},'WriteRowNames',true);

        data_sprob = cell2mat(ds.SmoothedProbabilities(offset,:));
        
        tab = [dates_str array2table(data_sprob,'VariableNames',strcat(repelem(ds.FirmNames,1,3),repmat({'_HV' '_MV' '_LV'},1,ds.N)))];
        writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{7},'WriteRowNames',true);
    end

    if (ds.RS4)
        if (sum([ds.RS2 ds.RS3 ds.RS4]) == 1)
            offset = 1;
        else
            sizes = cellfun(@(x)unique(size(x)),ds.TransitionMatrices(:,1));
            offset = find(sizes == 4,1,'first');
        end
        
        data_cmu = cell2mat(ds.ConditionalVariances(offset,:));

        tab = [dates_str array2table(data_cmu,'VariableNames',ds.FirmNames)];
        writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{8},'WriteRowNames',true);

        data_cs2 = cell2mat(ds.ConditionalVariances(offset,:));

        tab = [dates_str array2table(data_cs2,'VariableNames',ds.FirmNames)];
        writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{9},'WriteRowNames',true);

        data_sprob = cell2mat(ds.SmoothedProbabilities(offset,:));
        
        tab = [dates_str array2table(data_sprob,'VariableNames',strcat(repelem(ds.FirmNames,1,4),repmat({'_HV' '_HVC' '_LVC' '_LV'},1,ds.N)))];
        writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{10},'WriteRowNames',true);
    end
    
    if (ispc())
        try
            excel = actxserver('Excel.Application');
        catch
            return;
        end

        try
            exc_wb = excel.Workbooks.Open(out,0,false);

            if (~ds.RS2)
                exc_wb.Sheets.Item(ds.LabelsSheetsSimple{2}).Delete();
                exc_wb.Sheets.Item(ds.LabelsSheetsSimple{3}).Delete();
                exc_wb.Sheets.Item(ds.LabelsSheetsSimple{4}).Delete();
            end

            if (~ds.RS3)
                exc_wb.Sheets.Item(ds.LabelsSheetsSimple{5}).Delete();
                exc_wb.Sheets.Item(ds.LabelsSheetsSimple{6}).Delete();
                exc_wb.Sheets.Item(ds.LabelsSheetsSimple{7}).Delete();
            end

            if (~ds.RS4)
                exc_wb.Sheets.Item(ds.LabelsSheetsSimple{8}).Delete();
                exc_wb.Sheets.Item(ds.LabelsSheetsSimple{9}).Delete();
                exc_wb.Sheets.Item(ds.LabelsSheetsSimple{10}).Delete();
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

%% INTERNAL CALCULATIONS

function [mu_params,s2_params,p,sprob,dur,cmu,cs2,e] = regime_switching_2(r)

    [indep_s_params,~,s2_params,p,sprob,dur,cmu,cs2,e] = regime_switching(r,[],[],2,true,@finit,[],[]);
    mu_params = indep_s_params{1};

    function [x0,ai,bi,ae,be,lb,ub] = finit(dep,~,~,~,~,~,~,options)

        tol = 2 * options.TolCon;

        dep_s2 = var(dep) .* [1.5; 0.5];
        dep_mu = [min(mean(dep(dep < 0)),-tol); max(mean(dep(dep > 0)),tol)];

        x0 = [dep_s2; dep_mu];
        lb = [zeros(2,1); -Inf(2,1)];
        ub = [Inf(2,1); Inf(2,1)];

        ai = [-1 1 0 0; 0 0 1 -1];
        bi = ones(2,1) .* -tol;
        ae = [];
        be = [];

    end
    
end

function [mu_params,s2_params,p,sprob,dur,cmu,cs2,e] = regime_switching_3(r)

    tmm = [NaN NaN 0; NaN NaN NaN; 0 NaN NaN];

    [indep_s_params,~,s2_params,p,sprob,dur,cmu,cs2,e] = regime_switching(r,[],[],3,true,@finit,tmm,[]);
    mu_params = indep_s_params{1};

    function [x0,ai,bi,ae,be,lb,ub] = finit(dep,~,~,~,~,~,~,options)

        tol = 2 * options.TolCon;

        dep_mu = [min(mean(dep(dep < 0)),-tol); mean(dep); max(mean(dep(dep > 0)),tol)];
        
        if ((dep_mu(2) <= dep_mu(1)) || (dep_mu(2) >= dep_mu(3)))
            dep_mu(2) = mean([dep_mu(1) dep_mu(3)]);
        end
        
        dep_s2 = var(dep) .* (1.5:-0.5:0.5).';

        x0 = [dep_s2; dep_mu];
        lb = [zeros(3,1); -Inf(2,1); tol];
        ub = [Inf(3,1); 0; Inf(2,1)];

        ai = [-1 1 0 0 0 0; 0 -1 1 0 0 0; 0 0 0 1 -1 0; 0 0 0 0 1 -1];
        bi = ones(4,1) .* -tol;
        ae = [];
        be = [];

    end
    
end

function [mu_params,s2_params,p,sprob,dur,cmu,cs2,e] = regime_switching_4(r)

    tmm = [NaN NaN 0 NaN; NaN NaN 0 0; 0 0 NaN NaN; NaN 0 NaN NaN];

    [indep_s_params,~,s2_params,p,sprob,dur,cmu,cs2,e] = regime_switching(r,[],[],4,true,@finit,tmm,[]);
    mu_params = indep_s_params{1};

    function [x0,ai,bi,ae,be,lb,ub] = finit(dep,~,~,~,~,~,p0,options)

        tol = 2 * options.TolCon;

        [v,d] = eig(p0,'vector');
        d_sorted = sort(abs(d),'descend');
        rho = d_sorted(1);
        v_indices = (abs(d - rho) < (rho * 4 * eps())).';

        uprob = abs(v(:,v_indices).');
        uprob = uprob ./ sum(uprob,2);

        up12 = uprob(1) + uprob(2);
        up1 = uprob(1) / up12;
        up2 = uprob(2) / up12;

        up34 = uprob(3) + uprob(4);
        up3 = uprob(3) / up34;
        up4 = uprob(4) / up34;
        
        dep_s2 = var(dep) .* (2:-0.5:0.5).';

        dep_mu = repmat([min(mean(dep(dep < 0)),-tol); max(mean(dep(dep > 0)),tol)],2,1);
        dep_mu(1) = -1 * (abs(dep_mu(1)) .* 1.25);
        dep_mu(4) = dep_mu(4) .* 1.25;

        x0 = [dep_s2; dep_mu];
        lb = [zeros(4,1); -Inf; tol; -Inf; tol];
        ub = [Inf(4,1); 0; Inf; 0; Inf];

        ai = [-1 1 0 0 0 0 0 0; 0 0 -1 1 0 0 0 0; 0 0 0 0 up1 up2 0 0; 0 0 0 0 0 0 -up3 -up4];
        bi = ones(4,1) .* -tol;
        ae = [];
        be = [];

    end

end

%% PLOTTING

function plot_indicators(ds,target,id)

    if (strcmp(target,'RS2'))
        model = '(2 States)';
    elseif (strcmp(target,'RS3'))
        model = '(3 States)';
    else
        model = '(4 States)';
    end

    if (sum([ds.RS2 ds.RS3 ds.RS4]) == 1)
        ap = ds.AverageProbabilities;
        jp = ds.JointProbabilities;
    else
        if (strcmp(target,'RS2'))
            target = 2;
        elseif (strcmp(target,'RS3'))
            target = 3;
        else
            target = 4;
        end

        sizes = cellfun(@(x)unique(size(x)),ds.TransitionMatrices(:,1));
        offset = find(sizes == target,1,'first');
        
        ap = ds.AverageProbabilities(:,offset);
        jp = ds.JointProbabilities(:,offset);
    end
    
    ap = smooth_data(ap);
    jp = smooth_data(jp);

    f = figure('Name',['Regime-Switching Measures > Indicators ' model],'Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);
    
    if (all(ap == 0))
        sub_1 = subplot(2,1,1);
        area(ds.DatesNum,ones(ds.T,1),'EdgeColor','none','FaceColor',[0.8 0.8 0.8]);
        hold on;
            plot([ds.DatesNum(1) ds.DatesNum(end)],[0 1],'Color',[0 0 0],'LineWidth',1.5);
            plot([ds.DatesNum(1) ds.DatesNum(end)],[1 0],'Color',[0 0 0],'LineWidth',1.5);
        hold off;
        set(sub_1,'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTick',[]);
        set(sub_1,'YLim',[0 1],'YTick',[]);
    else
        sub_1 = subplot(2,1,1);
        plot(ds.DatesNum,ap);
        set(sub_1,'YLim',[0 1]);
        set(sub_1,'YTick',0:0.1:1,'YTickLabels',arrayfun(@(x)sprintf('%.f%%',x),(0:0.1:1) .* 100,'UniformOutput',false));
        set(sub_1,'XGrid','on','YGrid','on');
        
        if (ds.MonthlyTicks)
            date_ticks(sub_1,'x','mm/yyyy','KeepLimits','KeepTicks');
        else
            date_ticks(sub_1,'x','yyyy','KeepLimits');
        end
    end
    
    t1 = title(sub_1,'Average Probability of High Variance');
    set(t1,'Units','normalized');
    t1_position = get(t1,'Position');
    set(t1,'Position',[0.4783 t1_position(2) t1_position(3)]);
    
    if (all(jp == 0))
        sub_2 = subplot(2,1,2);
        area(ds.DatesNum,ones(ds.T,1),'EdgeColor','none','FaceColor',[0.8 0.8 0.8]);
        hold on;
            plot([ds.DatesNum(1) ds.DatesNum(end)],[0 1],'Color',[0 0 0],'LineWidth',1.5);
            plot([ds.DatesNum(1) ds.DatesNum(end)],[1 0],'Color',[0 0 0],'LineWidth',1.5);
        hold off;
        set(sub_2,'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTick',[]);
        set(sub_2,'YLim',[0 1],'YTick',[]);
    else
        sub_2 = subplot(2,1,2);
        plot(ds.DatesNum,jp);
        set(sub_2,'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
        set(sub_2,'YLim',plot_limits(jp,0,0));
        set(sub_2,'YTickLabels',arrayfun(@(x)sprintf('%.f%%',x),get(sub_2,'YTick') .* 100,'UniformOutput',false));
        set(sub_2,'XGrid','on','YGrid','on');
        
        if (ds.MonthlyTicks)
            date_ticks(sub_2,'x','mm/yyyy','KeepLimits','KeepTicks');
        else
            date_ticks(sub_2,'x','yyyy','KeepLimits');
        end
    end
    
    t2 = title(sub_2,'Joint Probability of High Variance');
    set(t2,'Units','normalized');
    t2_position = get(t2,'Position');
    set(t2,'Position',[0.4783 t2_position(2) t2_position(3)]);

    figure_title(['Indicators ' model]);

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_sequence(ds,target,id)

    n = ds.N;
    t = ds.T;
    dn = ds.DatesNum;
    mt = ds.MonthlyTicks;

    if (strcmp(target,'RS2'))
        k = 2;
        model = '2 States';
        states = {'HV' 'LV'};
    elseif (strcmp(target,'RS3'))
        k = 3;
        model = '3 States';
        states = {'HV' 'MV' 'LV'};
    else
        k = 4;
        model = '4 States';
        states = {'HV' 'HVC' 'LVC' 'LV'};
    end

    if (sum([ds.RS2 ds.RS3 ds.RS4]) == 1)
        offset = 1;
    else
        sizes = cellfun(@(x)unique(size(x)),ds.TransitionMatrices(:,1));
        offset = find(sizes == k,1,'first');
    end
    
    data_r = cell(1,n);
    
    for i = 1:n
        r_i = ds.Returns(:,i);
        
        r_i_m = mean(r_i,'omitnan');
        r_i_s = std(r_i,'omitnan');
        r_i((r_i < (r_i_m - (r_i_s * 3))) | (r_i > (r_i_m + (r_i_s * 3)))) = NaN;
        
        r_i_min = min(r_i,[],'omitnan');
        r_i_max = max(r_i,[],'omitnan');
        r_i = 0.25 + (0.5 .* ((r_i - r_i_min) ./ (r_i_max -r_i_min)));

        data_r{i} = r_i;
    end

    data_cs2 = mat2cell(smooth_data(cell2mat(ds.ConditionalVariances(offset,:))),t,ones(1,n));
    
    data_tm = ds.TransitionMatrices(offset,:);
    
    for i = 1:n
        data_tm{i} = flipud(data_tm{i});
    end
    
    data_dur = ds.Durations(offset,:);
    
    for i = 1:n
        data_dur_i = data_dur{i};
        data_dur{i} = data_dur_i ./ sum(data_dur_i);
    end

    data_sprob = cell2mat(ds.SmoothedProbabilities(offset,:));

    if (k == 2)
        data_sprob_hv = mat2cell(smooth_data(data_sprob(:,1:2:n*2)),t,ones(1,n));
    elseif (k == 3)
        data_sprob_hv = mat2cell(smooth_data(data_sprob(:,1:3:n*3)),t,ones(1,n));
    else
        data_sprob_hv = mat2cell(smooth_data(data_sprob(:,1:4:n*4) + data_sprob(:,2:4:n*4)),t,ones(1,n));
    end

    data = [repmat({dn},1,n); data_r; data_cs2; data_tm; data_dur; data_sprob_hv];
    
    plots_title = [repmat({'Conditional Variance'},1,n); repmat({'Transition Matrix'},1,n); repmat({'Probability of High Variance'},1,n); repmat({'Relative Durations'},1,n)];
    
    x_limits_dur = [0 (k + 1)];
    x_limits_tm = [1 (k + 1)];
    x_limits_ts = [dn(1) dn(end)];

    x_tick_dur = 1:k;
    x_tick_tm = 1.5:(k+0.5);
    
    x_tick_labels_small = states;
    
    all_cs2 = reshape(cell2mat(data(3,:)),n*t,1);
    all_cs2(isnan(all_cs2)) = [];
    y_limits_cs2 = plot_limits(all_cs2,0.1);
    y_limits_pc = [0 1];
    y_limits_tm = [1 (k + 1)];
    
    y_ticks_pc = 0:0.1:1;
    y_ticks_tm = 1.5:(k+0.5);
    
    y_tick_labels_pc = @(x)sprintf('%.f%%',x .* 100);
    y_tick_labels_tm = fliplr(states);
    
    core = struct();

    core.N = n;
    core.Data = data;
    core.Function = @(subs,data)plot_function(subs,data,k);

    core.OuterTitle = ['Regime-Switching Measures > ' model ' Time Series'];
    core.InnerTitle = [model ' Time Series'];
    core.SequenceTitles = ds.FirmNames;

    core.PlotsAllocation = [2 7];
    core.PlotsSpan = {(1:5) (6:7) (8:12) (13:14)};
    core.PlotsTitle = plots_title;

    core.XDates = {mt [] mt []};
    core.XGrid = {true false true false};
    core.XLabel = {[] [] [] []};
    core.XLimits = {x_limits_ts x_limits_tm x_limits_ts x_limits_dur};
    core.XRotation = {45 [] 45 []};
    core.XTick = {[] x_tick_tm [] x_tick_dur};
    core.XTickLabels = {[] x_tick_labels_small [] x_tick_labels_small};

    core.YGrid = {true false true true};
    core.YLabel = {[] [] [] []};
    core.YLimits = {y_limits_cs2 y_limits_tm y_limits_pc y_limits_pc};
    core.YRotation = {[] [] [] []};
    core.YTick = {[] y_ticks_tm y_ticks_pc y_ticks_pc};
    core.YTickLabels = {[] y_tick_labels_tm y_tick_labels_pc y_tick_labels_pc};

    sequential_plot(core,id);
    
    function plot_function(subs,data,k)
        
        x = data{1};
        r = data{2};
        cs2 = data{3};
        tm = data{4};
        dur = data{5};
        sprob_hv = data{6};

        d = find(isnan(cs2),1,'first');
        
        if (isempty(d))
            xd = [];
        else
            xd = x(d) - 1;
        end

        tmn = tm;
        tmn(tmn >= 0.5) = 1;
        tmn((tmn > 0) & (tmn < 0.5)) = 0.5;
        
        tmv = tm(:);
        
        [tm_x,tm_y] = meshgrid(1:k,1:k);
        tm_x = tm_x(:) + 0.5;
        tm_y = tm_y(:) + 0.5;
        tm_txt = cellstr(num2str(round(tmv .* 100,2),'~%.2f%%'));
        
        for j = 1:k^2
            tmv_j = tmv(j);
            
            if ((tmv_j == 0) || (tmv_j == 1))
                tm_txt{j} = strrep(tm_txt{j},'~','');
            end
        end
        
        tmnv = tmn(:);
        
        if (all(tmnv == 0))
            tm_cmap = [0.65 0.65 0.65];
        elseif (all(tmnv == 0.5))
            tm_cmap = [1 1 1];
        elseif (all(tmnv == 1))
            tm_cmap = [0.749 0.862 0.933];
        else
            tm_cmap = [0.65 0.65 0.65; 1 1 1; 0.749 0.862 0.933];
        end

        plot(subs(1),x,cs2,'Color',[0.000 0.447 0.741]);
        
        if (~isempty(xd))
            hold(subs(1),'on');
                plot(subs(1),[xd xd],get(subs(1),'YLim'),'Color',[1 0.4 0.4]);
            hold(subs(1),'off');
        end

        pcolor(subs(2),padarray(tmn,[1 1],'post'));
        colormap(subs(2),tm_cmap);
        text(subs(2),tm_x,tm_y,tm_txt,'FontSize',9 + 4 - k,'HorizontalAlignment','center');

        plot(subs(3),x,r,'Color',[0.65 0.65 0.65])
        hold(subs(3),'on');
            area(subs(3),x,sprob_hv,'EdgeColor',[0.000 0.447 0.741],'FaceAlpha',0.5,'FaceColor',[0.749 0.862 0.933]);
        hold(subs(3),'off');

        if (~isempty(xd))
            hold(subs(3),'on');
                plot(subs(3),[xd xd],get(subs(3),'YLim'),'Color',[1 0.4 0.4]);
            hold(subs(3),'off');
        end
        
        bar(subs(4),1:k,dur,'FaceColor',[0.749 0.862 0.933]);

    end

end

%% VALIDATION

function [rs2,rs3,rs4] = validate_booleans(rs2,rs3,rs4)

    if (~rs2 && ~rs3 && ~rs4)
        error('At least one regime-switching model must be computed.');
    end
    
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

    sheets = {'Indicators' 'RS2 CM' 'RS2 CV' 'RS2 SP' 'RS3 CM' 'RS3 CV' 'RS3 SP' 'RS4 CM' 'RS4 CV' 'RS4 SP'};
    
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
