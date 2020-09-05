% [INPUT]
% ds = A structure representing the dataset.
% temp = A string representing the full path to the Excel spreadsheet used as a template for the results file.
% out = A string representing the full path to the Excel spreadsheet to which the results are written, eventually replacing the previous ones.
% bw = An integer [21,252] representing the dimension of each rolling window (optional, default=252).
% op = A string (either 'BSM' for Black-Scholes-Merton or 'GC' for Gram-Charlier) representing the option pricing model (optional, default='BSM').
% lst = A float or a vector of floats (0,Inf) representing the long-term to short-term liabilities ratio(s) used to calculate D2C and D2D (optional, default=3).
% car = A float [0.03,0.20] representing the capital adequacy ratio used to calculate the D2C (optional, default=0.08).
% rr = A float [0,1] representing the recovery rate in case of default used to calculate the DIP (optional, default=0.45).
% f = An integer [2,n], where n is the number of firms, representing the number of systematic risk factors used to calculate the DIP (optional, default=2).
% l = A float [0.05,0.20] representing the importance sampling threshold used to calculate the DIP (optional, default=0.10).
% c = An integer [50,1000] representing the number of simulated samples used to calculate the DIP (optional, default=100).
% k = A float [0.90,0.99] representing the confidence level used by the Systemic CCA framework (optional, default=0.95).
% analyze = A boolean that indicates whether to analyse the results and display plots (optional, default=false).
%
% [OUTPUT]
% result = A structure representing the original dataset inclusive of intermediate and final calculations.
% stopped = A boolean that indicates whether the process has been stopped through user input.

function [result,stopped] = run_default(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('ds',@(x)validateattributes(x,{'struct'},{'nonempty'}));
        ip.addRequired('temp',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addRequired('out',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addOptional('bw',252,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 21 '<=' 252 'scalar'}));
        ip.addOptional('op','BSM',@(x)any(validatestring(x,{'BSM' 'GC'})));
        ip.addOptional('lst',3,@(x)validateattributes(x,{'double'},{'real' 'finite' '>' 0 'vector'}));
        ip.addOptional('car',0.08,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.03 '<=' 0.20 'scalar'}));
        ip.addOptional('rr',0.45,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0 '<=' 1 'scalar'}));
        ip.addOptional('f',2,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 2 'scalar'}));
        ip.addOptional('l',0.10,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.05 '<=' 0.20 'scalar'}));
        ip.addOptional('c',100,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 50 '<=' 1000 'scalar'}));
        ip.addOptional('k',0.95,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.90 '<=' 0.99 'scalar'}));
        ip.addOptional('analyze',false,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    ds = validate_dataset(ipr.ds,'default');
    temp = validate_template(ipr.temp);
    out = validate_output(ipr.out);
    bw = ipr.bw;
    op = ipr.op;
    lst = validate_lst(ipr.lst,ds.N);
    car = ipr.car;
    rr = ipr.rr;
    f = validate_f(ipr.f,ds.N);
    l = ipr.l;
    c = ipr.c;
    k = ipr.k;
    analyze = ipr.analyze;
    
    nargoutchk(1,2);
    
    [result,stopped] = run_default_internal(ds,temp,out,bw,op,lst,car,rr,f,l,c,k,analyze);

end

function [result,stopped] = run_default_internal(ds,temp,out,bw,op,lst,car,rr,f,l,c,k,analyze)

    result = [];
    stopped = false;
    e = [];

    ds = initialize(ds,bw,op,lst,car,rr,f,l,c,k);
    n = ds.N;
    t = ds.T;
    
    step_1 = 0.1;
    step_2 = 1 - step_1;

    rng(double(bitxor(uint16('T'),uint16('B'))));
    cleanup_1 = onCleanup(@()rng('default'));

    bar = waitbar(0,'Initializing default measures...','CreateCancelBtn',@(src,event)setappdata(gcbf(),'Stop',true));
    setappdata(bar,'Stop',false);
    cleanup_2 = onCleanup(@()delete(bar));
    
    pause(1);
    waitbar(0,bar,'Calculating default measures (step 1 of 2)...');
    pause(1);

    try

        r = max(0,ds.RiskFreeRate);

        firms_data = extract_firms_data(ds,{'Capitalizations' 'CDS' 'Liabilities'});
        
        futures(1:n) = parallel.FevalFuture;
        futures_max = 0;
        futures_results = cell(n,1);

        for i = 1:n
            offsets = [ds.Defaults(i) ds.Insolvencies(i)] - 1;
            futures(i) = parfeval(@main_loop_1,1,firms_data{i},offsets,r,ds.ST(i),ds.DT(i),ds.LCAR,ds.OP);
        end

        for i = 1:n
            if (getappdata(bar,'Stop'))
                stopped = true;
                break;
            end
            
            [future_index,value] = fetchNext(futures);
            futures_results{future_index} = value;
            
            futures_max = max([future_index futures_max]);
            waitbar(step_1 * ((futures_max - 1) / n),bar);

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
    waitbar(step_1,bar,'Finalizing default measures (step 1 of 2)...');
    pause(1);

    try
        ds = finalize_1(ds,futures_results);
    catch e
        delete(bar);
        rethrow(e);
    end
    
    pause(1);
    waitbar(step_1,bar,'Calculating default measures (step 2 of 2)...');
    pause(1);
    
    try

        r = distress_data(ds.Returns,ds.Insolvencies);
        windows_r = extract_rolling_windows(r,ds.BW);

        cds = distress_data(ds.CDS,ds.Insolvencies);
        lb = distress_data(ds.Liabilities,ds.Insolvencies);

        cl = ds.SCCACL;
        cl(isnan(cl)) = 0;
        windows_cl = extract_rolling_windows(cl,ds.BW);

        futures(1:t) = parallel.FevalFuture;
        futures_max = 0;
        futures_results = cell(t,1);

        for i = 1:t
            futures(i) = parfeval(@main_loop_2,1,windows_r{i},cds(i,:),lb(i,:),ds.LGD,ds.F,ds.L,ds.C,windows_cl{i},ds.K);
        end

        for i = 1:t
            if (getappdata(bar,'Stop'))
                stopped = true;
                break;
            end
            
            [future_index,value] = fetchNext(futures);
            futures_results{future_index} = value;
            
            futures_max = max([future_index futures_max]);
            waitbar(step_1 + (step_2 * ((futures_max - 1) / t)),bar);

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
    waitbar(1,bar,'Finalizing default measures (step 2 of 2)...');
    pause(1);

    try
        ds = finalize_2(ds,futures_results);
    catch e
        delete(bar);
        rethrow(e);
    end

    pause(1);
    waitbar(1,bar,'Writing default measures...');
    pause(1);
    
    try
        write_results(ds,temp,out);
        delete(bar);
    catch e
        delete(bar);
        rethrow(e);
    end
    
    if (analyze)
        safe_plot(@(id)plot_distances(ds,id));
        safe_plot(@(id)plot_sequence(ds,'D2D',true,id));
        safe_plot(@(id)plot_sequence(ds,'D2C',true,id));
        safe_plot(@(id)plot_dip(ds,id));
        safe_plot(@(id)plot_scca(ds,id));
        safe_plot(@(id)plot_sequence(ds,'SCCA EL',false,id));
        safe_plot(@(id)plot_sequence(ds,'SCCA CL',false,id));
        safe_plot(@(id)plot_rankings(ds,id));
    end
    
    result = ds;

end

%% DATA

function ds = initialize(ds,bw,op,lst,car,rr,f,l,c,k)

    q = [(0.900:0.025:0.975) 0.99];
    q = q(q >= k);

    n = ds.N;
    t = ds.T;

    ds.A = 1 - k;
    ds.BW = bw;
    ds.C = c;
    ds.CAR = car;
    ds.DT = max(0.5,0.7 - (0.3 .* (1 ./ lst)));
    ds.F = f;
    ds.K = k;
    ds.L = l;
    ds.LCAR = 1 / (1 - car);
    ds.LGD = 1 - rr;
    ds.LST = lst;
    ds.OP = op;
    ds.RR = rr;
    ds.ST =  1 ./ (1 + lst);

    op_label =  [' (' ds.OP ')'];
    d2c_label =  [' (' ds.OP ', CAR=' num2str(ds.CAR * 100) ')'];
    dip_label =  [' (RR=' num2str(ds.RR * 100) ', F=' num2str(ds.F) ', L=' num2str(ds.L * 100) ')'];
    scca_label = [' (' ds.OP ', K=' num2str(ds.K * 100) ')'];

    ds.LabelsMeasuresSimple = {'D2D' 'D2C' 'SCCA EL' 'SCCA CL'};
    ds.LabelsMeasures = {['D2D' op_label] ['D2C' d2c_label] ['SCCA EL' op_label] ['SCCA CL' op_label]};
    
    ds.LabelsIndicatorsSimple = {'Average D2D' 'Average D2C' 'Portfolio D2D' 'Portfolio D2C' 'DIP' 'SCCA JES'};
	ds.LabelsIndicatorsShort = {'AD2D' 'AD2C' 'PD2D' 'PD2C' 'DIP' 'SCCAJES'};
    ds.LabelsIndicators = {['Average D2D' op_label] ['Average D2C' d2c_label] ['Portfolio D2D' op_label] ['Portfolio D2C' d2c_label] ['DIP' dip_label] ['SCCA JES' scca_label]};

    ds.LabelsSheetsSimple = [ds.LabelsMeasuresSimple {'Indicators'}];
    ds.LabelsSheets = [ds.LabelsMeasures {'Indicators'}];

    ds.D2D = NaN(t,n);
    ds.D2C = NaN(t,n);

    ds.SCCAAlphas = NaN(t,n);
    ds.SCCAEL = NaN(t,n);
    ds.SCCACL = NaN(t,n);
    ds.SCCAJointVaRs = NaN(t,numel(q));

    ds.Indicators = NaN(t,numel(ds.LabelsIndicators));
    
    ds.ComparisonReferences = {'Indicators' [] strcat({'DE-'},ds.LabelsIndicatorsShort)};

end

function ds = finalize_1(ds,results)
  
    n = ds.N;

    for i = 1:n
        result = results{i};

        ds.D2D(1:result.Offset1,i) = result.D2D;
        ds.D2C(1:result.Offset1,i) = result.D2C;

        ds.SCCAAlphas(1:result.Offset2,i) = result.SCCAAlphas;
        ds.SCCAEL(1:result.Offset2,i) = result.SCCAEL;
        ds.SCCACL(1:result.Offset2,i) = result.SCCACL;
    end

    [d2d_avg,d2c_avg,d2d_por,d2c_por] = calculate_overall_distances(ds);
    ds.Indicators(:,1) = d2d_avg;
    ds.Indicators(:,2) = d2c_avg;
    ds.Indicators(:,3) = d2d_por;
    ds.Indicators(:,4) = d2c_por;
    
    [rc,rs] = kendall_rankings(ds,ds.LabelsMeasuresSimple);
    ds.RankingConcordance = rc;
    ds.RankingStability = rs;

end

function ds = finalize_2(ds,results)

    t = ds.T;

    for i = 1:t
        result = results{i};
        
        ds.SCCAJointVaRs(i,:) = result.SCCAJointVaRs;

        ds.Indicators(i,5) = result.DIP;
        ds.Indicators(i,6) = result.SCCAJointES;
    end
    
    w = max(round(nthroot(ds.BW,1.81),0),5); 
    ds.Indicators(:,5) = sanitize_data(ds.Indicators(:,5),ds.DatesNum,w,[]);

end

function lst = validate_lst(lst,n)

    if (isscalar(lst))
        lst = ones(1,n) .* lst;
    else
        if (numel(lst) ~= n)
            error(['The number of lst coefficients, when specified as a vector, must be equal to the number of firms (' num2str(n) ').']);
        end
        
        lst = lst(:).';
    end

end

function out_file = validate_output(out_file)

    [path,name,extension] = fileparts(out_file);

    if (~strcmp(extension,'.xlsx'))
        out_file = fullfile(path,[name extension '.xlsx']);
    end
    
end

function f = validate_f(f,n)

    if (f > n)
        error(['The value of ''f'' is invalid. Expected input to be less than or equal to (' num2str(n) ').']);
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

    sheets = {'D2D' 'D2C' 'SCCA EL' 'SCCA CL' 'Indicators'};

    if (~all(ismember(sheets,file_sheets)))
        error(['The template must contain the following sheets: ' sheets{1} sprintf(', %s',sheets{2:end}) '.']);
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
        error('The output file could not be created from the template file.');
    end

    dates_str = cell2table(ds.DatesStr,'VariableNames',{'Date'});

    for i = 1:(numel(ds.LabelsSheetsSimple) - 1)
        sheet = ds.LabelsSheetsSimple{i};
        measure = strrep(sheet,' ','');

        tab = [dates_str array2table(ds.(measure),'VariableNames',ds.FirmNames)];
        writetable(tab,out,'FileType','spreadsheet','Sheet',sheet,'WriteRowNames',true);
    end

    tab = [dates_str array2table(ds.Indicators,'VariableNames',strrep(ds.LabelsIndicatorsSimple,' ','_'))];
    writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{end},'WriteRowNames',true);    

    if (ispc())
        try
            excel = actxserver('Excel.Application');
        catch
            return;
        end

        try
            exc_wb = excel.Workbooks.Open(out,0,false);

            for i = 1:numel(ds.LabelsSheet)
                exc_wb.Sheets.Item(ds.LabelsSheetSimple{i}).Name = ds.LabelsSheet{i};
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

function window_results = main_loop_1(firm_data,offsets,r,st,dt,lcar,op)

    t = size(firm_data,1);

    offset1 = min(offsets(1),t);
    cp_o = max(1e-6,firm_data(1:offset1,1));
    lb_o = max(1e-6,firm_data(1:offset1,3));
    db_o = (lb_o .* st) + (dt .* (lb_o .* (1 - st)));
    r_o = r(1:offset1);

    [va,vap] = kmv_structural(cp_o,db_o,r_o,1,op);
    [d2d,d2c] = calculate_distances(va,vap,db_o,r_o,1,lcar);
    
    offset2 = min(min(offsets),t);
    cp_o = max(1e-6,firm_data(1:offset2,1));
    lb_o = max(1e-6,firm_data(1:offset2,3));
    db_o = (lb_o .* st) + (dt .* (lb_o .* (1 - st)));
    r_o = r(1:offset2);
    cds_o = firm_data(1:offset2,2);

    [va,vap] = kmv_structural(cp_o,db_o,r_o,1,op);
    [el,cl,a] = contingent_claims_analysis(va,vap,cds_o,db_o,r_o,1);

    window_results = struct();
    window_results.Offset1 = offset1;
    window_results.Offset2 = offset2;
    window_results.D2D = d2d;
    window_results.D2C = d2c;
    window_results.SCCAAlphas = a;
    window_results.SCCAEL = el;
    window_results.SCCACL = cl;

end

function window_results = main_loop_2(r,cds,lb,lgd,f,l,c,cl,k)

    window_results = struct();

    dip = distress_insurance_premium(r,cds,lb,lgd,f,l,c);
    window_results.DIP = dip;

    [joint_vars,joint_es] = mgev_joint_risks(cl,k);
    window_results.SCCAJointVaRs = joint_vars;
    window_results.SCCAJointES = joint_es;

end

function [d2d,d2c] = calculate_distances(va,vap,db,r,t,lcar)

    s = vap(1);
    rst = (r + (0.5 * s^2)) * t;
    st = s * sqrt(t);

    d1 = (log(va ./ db) + rst) ./ st;
    d2d = d1 - st;

    d1 = (log(va ./ (lcar .* db)) + rst) ./ st;
    d2c = d1 - st;

end

function [d2d_avg,d2c_avg,d2d_por,d2c_por] = calculate_overall_distances(ds)

    n = ds.N;
    t = ds.T;

    cp = ds.Capitalizations;
    lb = ds.Liabilities;
    r = max(0,ds.RiskFreeRate);

    weights = cp ./ repmat(sum(cp,2,'omitnan'),1,n);
    d2d_avg = sum(ds.D2D .* weights,2,'omitnan');
    d2c_avg = sum(ds.D2C .* weights,2,'omitnan');

    cp = max(1e-6,sum(cp,2,'omitnan'));
    lbs = lb .* repmat(ds.ST,t,1);
    lbl = repmat(ds.DT,t,1) .* (lb .* (1 - repmat(ds.ST,t,1)));
    db = max(1e-6,sum(lbs + lbl,2,'omitnan'));

    [va,vap] = kmv_structural(cp,db,r,1,ds.OP);
    [d2d_por,d2c_por] = calculate_distances(va,vap,db,r,1,ds.LCAR);
    
end

%% PLOTTING

function plot_distances(ds,id)

    distances = ds.Indicators(:,1:4);

    y_min = min(min(min(distances)),-1);
    y_max = max(max(distances));
    y_limits = plot_limits(distances,0.1,[],[],-1);
    
    y_ticks = floor(y_min):0.5:ceil(y_max);
    y_ticks_labels = arrayfun(@(x)sprintf('%.1f',x),y_ticks,'UniformOutput',false);

    f = figure('Name','Default Measures > Distances','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    sub_1 = subplot(2,2,1);
    plot(sub_1,ds.DatesNum,smooth_data(distances(:,1)),'Color',[0.000 0.447 0.741]);
    hold on;
        p = plot(sub_1,ds.DatesNum,zeros(ds.T,1),'Color',[1 0.4 0.4]);
    hold off;
    title(sub_1,ds.LabelsIndicators{1});
    
    sub_2 = subplot(2,2,2);
    plot(sub_2,ds.DatesNum,smooth_data(distances(:,3)),'Color',[0.000 0.447 0.741]);
    hold on;
        plot(sub_2,ds.DatesNum,zeros(ds.T,1),'Color',[1 0.4 0.4]);
    hold off;
    title(sub_2,ds.LabelsIndicators{2});
    
    sub_3 = subplot(2,2,3);
    plot(sub_3,ds.DatesNum,smooth_data(distances(:,2)),'Color',[0.000 0.447 0.741]);
    hold on;
        plot(sub_3,ds.DatesNum,zeros(ds.T,1),'Color',[1 0.4 0.4]);
    hold off;
    title(sub_3,ds.LabelsIndicators{3});
    
    sub_4 = subplot(2,2,4);
    plot(sub_4,ds.DatesNum,smooth_data(distances(:,4)),'Color',[0.000 0.447 0.741]);
    hold on;
        plot(sub_4,ds.DatesNum,zeros(ds.T,1),'Color',[1 0.4 0.4]);
    hold off;
    title(sub_4,ds.LabelsIndicators{4});
    
    set([sub_1 sub_2 sub_3 sub_4],'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
    set([sub_1 sub_2 sub_3 sub_4],'YLim',y_limits,'YTick',y_ticks,'YTickLabel',y_ticks_labels);
    set([sub_1 sub_2 sub_3 sub_4],'XGrid','on','YGrid','on');
    
    if (ds.MonthlyTicks)
        date_ticks([sub_1 sub_2 sub_3 sub_4],'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        date_ticks([sub_1 sub_2 sub_3 sub_4],'x','yyyy','KeepLimits');
    end

    l = legend(sub_1,p,'Default Threshold','Location','best');
    set(l,'Units','normalized');
    l_position = get(l,'Position');
    set(l,'Position',[0.4683 0.4799 l_position(3) l_position(4)]);

    figure_title('Distances');

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_dip(ds,id)

    dip = ds.Indicators(:,5);
    y = smooth_data(dip);

    f = figure('Name','Default Measures > Distress Insurance Premium','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    sub_1 = subplot(1,6,1:5);
    plot(sub_1,ds.DatesNum,y,'Color',[0.000 0.447 0.741]);
    set(sub_1,'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
    set(sub_1,'XGrid','on','YGrid','on');
    title(sub_1,ds.LabelsIndicators{5});
    
    if (ds.MonthlyTicks)
        date_ticks(sub_1,'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        date_ticks(sub_1,'x','yyyy','KeepLimits');
    end
    
    sub_2 = subplot(1,6,6);
    boxplot(sub_2,y,'Notch','on','Symbol','k.');
    set(findobj(f,'type','line','Tag','Median'),'Color','g');
    set(findobj(f,'-regexp','Tag','\w*Whisker'),'LineStyle','-');
    set(sub_2,'TickLength',[0 0],'XTick',[],'XTickLabels',[]);

    figure_title('Distress Insurance Premium');
    
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

    f = figure('Name','Default Measures > Rankings','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

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
    set(sub_2,'YDir','reverse','YTick',off,'YTickLabels',labels,'YTickLabelRotation',45);
    title(sub_2,'Ranking Concordance');

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

function plot_scca(ds,id)

    el = sum(ds.SCCAEL,2,'omitnan');
    cl = sum(ds.SCCACL,2,'omitnan');
    alpha = cl ./ el;
    jes = ds.Indicators(:,6);

    f = figure('Name','Default Measures > Systemic CCA','Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    sub_1 = subplot(2,2,[1 2]);
    a1 = area(sub_1,ds.DatesNum,smooth_data(el),'EdgeColor','none','FaceColor',[0.65 0.65 0.65]);
    hold on;
        p1 = plot(sub_1,ds.DatesNum,smooth_data(cl),'Color',[0.000 0.447 0.741]);
    hold off;
    legend(sub_1,[a1 p1],'Expected Losses','Contingent Liabilities','Location','best');
    t1 = title(sub_1,'Values');
    set(t1,'Units','normalized');
    t1_position = get(t1,'Position');
    set(t1,'Position',[0.4783 t1_position(2) t1_position(3)]);
    
    sub_2 = subplot(2,2,3);
    plot(sub_2,ds.DatesNum,smooth_data(alpha),'Color',[0.000 0.447 0.741]);
    title(sub_2,'Average Alpha');
    
    sub_3 = subplot(2,2,4);
    plot(sub_3,ds.DatesNum,smooth_data(jes),'Color',[0.000 0.447 0.741]);
    title(sub_3,['Joint ES (K=' sprintf('%.1f%%',(ds.K * 100)) ')']);
    
    set([sub_1 sub_2 sub_3],'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
    set([sub_2 sub_3],'XGrid','on','YGrid','on');

    if (ds.MonthlyTicks)
        date_ticks([sub_1 sub_2 sub_3],'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        date_ticks([sub_1 sub_2 sub_3],'x','yyyy','KeepLimits');
    end

    figure_title(['Systemic CCA (' ds.OP ')']);

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function plot_sequence(ds,target,distance,id)

    n = ds.N;
    t = ds.T;
    dn = ds.DatesNum;
    mt = ds.MonthlyTicks;
    
    if (distance)
        ts = smooth_data(ds.(strrep(target,' ','')));
    else
        ts = smooth_data(ds.(strrep(target,' ','')));
    end
    
    data = [repmat({dn},1,n); mat2cell(ts,t,ones(1,n))];

    [~,index] = ismember(target,ds.LabelsMeasuresSimple);
    plots_title = repmat(ds.LabelsMeasures(index),1,n);
    
    x_limits = [dn(1) dn(end)];

    if (distance)
        y_limits = plot_limits(ts,0.1,[],[],-1);
    else
        y_limits = plot_limits(ts,0.1);
    end

    core = struct();

    core.N = n;
    core.Data = data;
    core.Function = @(subs,data)plot_function(subs,data,distance);

    core.OuterTitle = ['Default Measures > ' target ' Time Series'];
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
    
    function plot_function(subs,data,distance)

        x = data{1};
        y = data{2};
        
        d = find(isnan(y),1,'first');
        
        if (isempty(d))
            xd = [];
        else
            xd = x(d) - 1;
        end
        
        plot(subs(1),x,y,'Color',[0.000 0.447 0.741]);
        
        if (distance)
            hold(subs(1),'on');
                plot(subs(1),x,zeros(numel(x),1),'Color',[1 0.4 0.4]);
            hold(subs(1),'off');
        end

        if (~isempty(xd))
            hold(subs(1),'on');
                plot(subs(1),[xd xd],get(subs(1),'YLim'),'Color',[1 0.4 0.4]);
            hold(subs(1),'off');
        end

    end

end
