% [INPUT]
% ds = A structure representing the dataset.
% temp = A string representing the full path to the Excel spreadsheet used as a template for the results file.
% out = A string representing the full path to the Excel spreadsheet to which the results are written, eventually replacing the previous ones.
% bw = An integer [21,252] representing the dimension of each rolling window (optional, default=252).
% a = A float [0.01,0.10] representing the target quantile (optional, default=0.05).
% lags = An integer [10,60] representing the maximum number of lags (optional, default=60).
% cim = A string representing the computational approach of confidence intervals (optional, default='SB'):
%   - 'SB' for stationary bootstrap.
%   - 'SN' for self-normalization.
% cis = Optional argument representing the significance level of confidence intervals and whose value depends on the the chosen computational approach:
%   - for stationary bootstrap cross-quantilograms, a float (0.0,0.1] (default=0.050);
%   - for self-normalization cross-quantilograms, a float {0.005;0.010;0.025;0.050;0.100} (default=0.050).
% cip = Optional argument whose type depends on the chosen computational approach:
%   - for stationary bootstrap cross-quantilograms, an integer [10,1000] representing the number of bootstrap iterations of confidence intervals (default=100);
%   - for self-normalization cross-quantilograms, a float {0.00;0.01;0.03;0.05;0.10;0.15;0.20;0.30} representing the minimum subsample size, as a fraction, for confidence intervals (default=0.10).
% analyze = A boolean that indicates whether to analyse the results and display plots (optional, default=false).
%
% [OUTPUT]
% result = A structure representing the original dataset inclusive of intermediate and final calculations.
% stopped = A boolean that indicates whether the process has been stopped through user input.

function [result,stopped] = run_cross_quantilogram(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('ds',@(x)validateattributes(x,{'struct'},{'nonempty'}));
        ip.addRequired('temp',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addRequired('out',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addOptional('bw',252,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 21 '<=' 252 'scalar'}));
        ip.addOptional('a',0.05,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.01 '<=' 0.10 'scalar'}));
        ip.addOptional('lags',60,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 10 '<=' 60 'scalar'}));
        ip.addOptional('cim','SB',@(x)any(validatestring(x,{'SB' 'SN'})));
        ip.addOptional('cis',0.050,@(x)validateattributes(x,{'double'},{'real' 'finite'}));
        ip.addOptional('cip',[],@(x)validateattributes(x,{'double'},{'real' 'finite'}));
        ip.addOptional('analyze',false,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    ds = validate_dataset(ipr.ds,'cross-quantilogram');
    temp = validate_template(ipr.temp);
    out = validate_output(ipr.out);
    bw = ipr.bw;
    a = ipr.a;
    lags = ipr.lags;
    [cim,cis,cip] = validate_ci(ipr.cim,ipr.cis,ipr.cip);
    analyze = ipr.analyze;
    
    nargoutchk(1,2);

    [result,stopped] = run_cross_quantilogram_internal(ds,temp,out,bw,a,lags,cim,cis,cip,analyze);

end

function [result,stopped] = run_cross_quantilogram_internal(ds,temp,out,bw,a,lags,cim,cis,cip,analyze)

    result = [];
    stopped = false;
    e = [];

    ds = initialize(ds,bw,a,lags,cim,cis,cip);
    n = ds.N;
    t = ds.T;
    
    rng(double(bitxor(uint16('T'),uint16('B'))));
    cleanup_1 = onCleanup(@()rng('default'));

    bar = waitbar(0,'Initializing cross-quantilogram measures...','CreateCancelBtn',@(src,event)setappdata(gcbf(),'Stop', true));
    setappdata(bar,'Stop',false);
    cleanup_2 = onCleanup(@()delete(bar));
    
    pause(1);
    waitbar(0,bar,'Calculating cross-quantilogram measures...');
    pause(1);

    try

        idx = ds.Index;
        r = ds.Returns;

        futures(1:n) = parallel.FevalFuture;
        futures_max = 0;
        futures_results = cell(n,1);

        if (ds.PI)
            sv = ds.StateVariables;
            
            for i = 1:n
                offset = min(ds.Defaults(i) - 1,t);
                idx_i = idx(1:offset);
                r_i = r(1:offset,i);
                sv_i = sv(1:offset,:);

                futures(i) = parfeval(@main_loop,1,idx_i,r_i,sv_i,ds.A,ds.Lags,ds.CIM,ds.CIS,ds.CIP);
            end
        else
            for i = 1:n
                offset = min(ds.Defaults(i) - 1,t);
                idx_i = idx(1:offset);
                r_i = r(1:offset,i);

                futures(i) = parfeval(@main_loop,1,idx_i,r_i,[],ds.A,ds.Lags,ds.CIM,ds.CIS,ds.CIP);
            end
        end

        for i = 1:n
            if (getappdata(bar,'Stop'))
                stopped = true;
                break;
            end
            
            [future_index,value] = fetchNext(futures);
            futures_results{future_index} = value;
            
            futures_max = max([future_index futures_max]);
            waitbar((futures_max - 1) / n,bar);

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
    waitbar(1,bar,'Finalizing cross-quantilogram measures...');
    pause(1);

    try
        ds = finalize(ds,futures_results);
    catch e
        delete(bar);
        rethrow(e);
    end

    pause(1);
    waitbar(1,bar,'Writing cross-quantilogram measures...');
    pause(1);
    
    try
        write_results(ds,temp,out);
        delete(bar);
    catch e
        delete(bar);
        rethrow(e);
    end
    
    if (analyze)
        safe_plot(@(id)plot_sequence(ds,'Full',id));
        
        if (ds.PI)
            safe_plot(@(id)plot_sequence(ds,'Partial',id));
        end
    end
    
    result = ds;

end

%% PROCESS

function ds = initialize(ds,bw,a,lags,cim,cis,cip)

    n = ds.N;

    ds.A = a;
    ds.BW = bw;
    ds.CIM = cim;
    ds.CIP = cip;
    ds.CIS = cis;
    ds.Lags = lags;
    ds.PI = ~isempty(ds.StateVariables);
    
    label = [' (' ds.CIM ', ' num2str(ds.CIS) ', P=' num2str(ds.CIP) ')'];

    ds.LabelsMeasuresSimple = {'Full From' 'Full To'};
    ds.LabelsMeasures = {['Full From' label] ['Full To' label]};
    
    ds.LabelsSheetsSimple = ds.LabelsMeasuresSimple;
    ds.LabelsSheets = ds.LabelsMeasures;
    
    ds.CQFullFrom = NaN(lags,n,3);
    ds.CQFullTo = NaN(lags,n,3);

    if (ds.PI)
        ds.LabelsMeasuresSimple = [ds.LabelsMeasuresSimple {'Partial From' 'Partial To'}];
        ds.LabelsMeasures = [ds.LabelsMeasures {['Partial From' label] ['Partial To' label]}];
        
        ds.LabelsSheetsSimple = ds.LabelsMeasuresSimple;
        ds.LabelsSheets = ds.LabelsMeasures;
        
        ds.CQPartialFrom = NaN(lags,n,3);
        ds.CQPartialTo = NaN(lags,n,3);
    end
    
    ds.ComparisonReferences = {};

end

function window_results = main_loop(idx,r,sv,a,lags,cim,cis,cip)

    window_results = struct();

    from = [r idx];
    to = [idx r];
    cq_full = zeros(lags,2,3);

    if (strcmp(cim,'SB'))
        for k = 1:lags
            [cq,ci] = cross_quantilograms_sb(from,a,k,cis,cip);
            cq_full(k,1,:) = [cq ci];

            [cq,ci] = cross_quantilograms_sb(to,a,k,cis,cip);
            cq_full(k,2,:) = [cq ci];
        end
    else
        for k = 1:lags
            [cq,ci] = cross_quantilograms_sn(from,a,k,cis,cip);
            cq_full(k,1,:) = [cq ci];

            [cq,ci] = cross_quantilograms_sn(to,a,k,cis,cip);
            cq_full(k,2,:) = [cq ci];
        end
    end
    
    window_results.CQFull = cq_full;
    
    if (~isempty(sv))
        from = [r idx sv];
        to = [idx r sv];
        cq_partial = zeros(lags,2,3);

        if (strcmp(cim,'SB'))
            for k = 1:lags
                [cq,ci] = cross_quantilograms_sb(from,a,k,cis,cip);
                cq_partial(k,1,:) = [cq ci];

                [cq,ci] = cross_quantilograms_sb(to,a,k,cis,cip);
                cq_partial(k,2,:) = [cq ci];
            end
        else
            for k = 1:lags
                [cq,ci] = cross_quantilograms_sn(from,a,k,cis,cip);
                cq_partial(k,1,:) = [cq ci];

                [cq,ci] = cross_quantilograms_sn(to,a,k,cis,cip);
                cq_partial(k,2,:) = [cq ci];
            end
        end
        
        window_results.CQPartial = cq_partial;
    end

end

function ds = finalize(ds,results)
  
    n = ds.N;

    for i = 1:n
        result = results{i};
        
        for j = 1:3
            ds.CQFullFrom(:,i,j) = result.CQFull(:,1,j);
            ds.CQFullTo(:,i,j) = result.CQFull(:,2,j);
        end
    end
    
    if (ds.PI)
        for i = 1:n
            result = results{i};

            for j = 1:3
                ds.CQPartialFrom(:,i,j) = result.CQPartial(:,1,j);
                ds.CQPartialTo(:,i,j) = result.CQPartial(:,2,j);
            end
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
    
    lags = num2cell(repelem(1:ds.Lags,1,3).');
    labels = repmat({'CQ' 'Lower CI' 'Upper CI'},1,ds.Lags).';
    row_headers = [labels lags];

    vars = [row_headers num2cell(reshape(permute(ds.CQFullFrom,[3 1 2]),ds.Lags * 3,ds.N))];
    tab = cell2table(vars,'VariableNames',[{'Value' 'Lag'} ds.FirmNames]);
    writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{1},'WriteRowNames',true);
    
    vars = [row_headers num2cell(reshape(permute(ds.CQFullTo,[3 1 2]),ds.Lags * 3,ds.N))];
    tab = cell2table(vars,'VariableNames',[{'Value' 'Lag'} ds.FirmNames]);
    writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{2},'WriteRowNames',true);
    
    if (ds.PI)
        vars = [row_headers num2cell(reshape(permute(ds.CQPartialFrom,[3 1 2]),ds.Lags * 3,ds.N))];
        tab = cell2table(vars,'VariableNames',[{'Value' 'Lag'} ds.FirmNames]);
        writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{3},'WriteRowNames',true);

        vars = [row_headers num2cell(reshape(permute(ds.CQPartialTo,[3 1 2]),ds.Lags * 3,ds.N))];
        tab = cell2table(vars,'VariableNames',[{'Value' 'Lag'} ds.FirmNames]);
        writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{4},'WriteRowNames',true);
    else
        if (ispc())
            try
                excel = actxserver('Excel.Application');
            catch
                return;
            end

            try
                exc_wb = excel.Workbooks.Open(out,0,false);

                exc_wb.Sheets.Item('Partial From').Delete();
                exc_wb.Sheets.Item('Partial To').Delete();

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
    
    if (ispc())
        try
            excel = actxserver('Excel.Application');
        catch
            return;
        end

        try
            exc_wb = excel.Workbooks.Open(out,0,false);
            
            if (ds.PI)
                offset = 4;
            else
                offset = 2;
            end

            for i = 1:offset
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

function plot_sequence(ds,target,id)

    n = ds.N;
    lags = ds.Lags;
    cq_from_all = ds.(['CQ' target 'From']);
    cq_to_all = ds.(['CQ' target 'To']);
    
    data = [repmat({(1:lags).'},1,n); cell(6,n)];
	
    if (strcmp(target,'Full'))
        plots_title = [repmat(ds.LabelsMeasures(1),1,n); repmat(ds.LabelsMeasures(2),1,n)];
    else
        plots_title = [repmat(ds.LabelsMeasures(3),1,n); repmat(ds.LabelsMeasures(4),1,n)];
    end
    
    for i = 1:n
        for j = 2:4
            data{j,i} = cq_from_all(:,i,j-1);
        end
        
        for j = 5:7
            data{j,i} = cq_to_all(:,i,j-4);
        end
    end

    x_limits = [0.3 (lags + 0.7)];
    
    if (lags <= 10)
        x_tick = 1:lags;
    elseif (lags <= 20)
        x_tick = [1 5:5:lags];
    else
        x_tick = [1 10:10:lags];
    end

    if (ds.PI)
        y_limits = plot_limits([ds.CQFullFrom(:) ds.CQFullTo(:) ds.CQPartialFrom(:) ds.CQPartialTo(:)],0.1);
    else
        y_limits = plot_limits([ds.CQFullFrom(:) ds.CQFullTo(:)],0.1);
    end
    
    y_tick_labels = @(x)sprintf('%.2f',x);

    core = struct();

    core.N = n;
    core.Data = data;
    core.Function = @(subs,data)plot_function(subs,data);

    core.OuterTitle = ['Cross-Quantilogram Measures > ' target ' Cross-Quantilograms'];
    core.InnerTitle = [target ' Cross-Quantilograms'];
    core.SequenceTitles = ds.FirmNames;

    core.PlotsAllocation = [2 1];
    core.PlotsSpan = {1 2};
    core.PlotsTitle = plots_title;

    core.XDates = {[] []};
    core.XGrid = {false false};
    core.XLabel = {[] []};
    core.XLimits = {x_limits x_limits};
    core.XRotation = {[] []};
    core.XTick = {x_tick x_tick};
    core.XTickLabels = {[] []};

    core.YGrid = {false false};
    core.YLabel = {[] []};
    core.YLimits = {y_limits y_limits};
    core.YRotation = {[] []};
    core.YTick = {[] []};
    core.YTickLabels = {y_tick_labels y_tick_labels};

    sequential_plot(core,id);
    
    function plot_function(subs,data)
        
        x = data{1};
        cq_from = data{2};
        cq_from_cl = data{3};
        cq_from_ch = data{4};
        cq_to = data{5};
        cq_to_cl = data{6};
        cq_to_ch = data{7};

        bar(subs(1),x,cq_from,0.6,'EdgeColor','none','FaceColor',[0.749 0.862 0.933]);
        hold(subs(1),'on');
            plot(subs(1),x,cq_from_cl,'Color',[1 0.4 0.4],'LineWidth',1);
            plot(subs(1),x,cq_from_ch,'Color',[1 0.4 0.4],'LineWidth',1);
        hold(subs(1),'off');
        
        bar(subs(2),x,cq_to,0.6,'EdgeColor','none','FaceColor',[0.749 0.862 0.933]);
        hold(subs(2),'on');
            plot(subs(2),x,cq_to_cl,'Color',[1 0.4 0.4],'LineWidth',1);
            plot(subs(2),x,cq_to_ch,'Color',[1 0.4 0.4],'LineWidth',1);
        hold(subs(2),'off');
        
    end

end

%% VALIDATION

function [cim,cis,cip] = validate_ci(cim,cis,cip)

    if (strcmp(cim,'SB'))
        validateattributes(cis,{'double'},{'real' 'finite' '>' 0 '<=' 0.1 'scalar'});
        
        if (isempty(cip))
            cip = 100;
        else
            validateattributes(cip,{'double'},{'real' 'finite' 'integer' '>=' 10 '<=' 1000 'scalar'});
        end
    else
        validateattributes(cis,{'double'},{'real' 'finite' 'scalar'});
        cis_allowed = [0.005 0.010 0.025 0.050 0.100];

        if (~ismember(cis,cis_allowed))
            cis_allowed_text = [sprintf('%.3f',cis_allowed(1)) sprintf(', %.3f',cis_allowed(2:end))];
            error(['The value of ''cis'' is invalid. Expected input to have one of the following values: ' cis_allowed_text '.']);
        end
        
        if (isempty(cip))
            cip = 0.10;
        else
            validateattributes(cip,{'double'},{'real' 'finite' 'scalar'});
            cip_allowed = [0.00 0.01 0.03 0.05 0.10 0.15 0.20 0.30];

            if (~ismember(cip,cip_allowed))
                cip_allowed_text = [sprintf('%.2f',cip_allowed(1)) sprintf(', %.2f',cip_allowed(2:end))];
                error(['The value of ''cip'' is invalid. Expected input to have one of the following values: ' cip_allowed_text '.']);
            end
        end
    end
  
end

function out = validate_output(out)

    [path,name,extension] = fileparts(out);

    if (~strcmp(extension,'.xlsx'))
        out = fullfile(path,[name extension '.xlsx']);
    end
    
end

function temp = validate_template(temp)

    sheets = {'Full From' 'Full To' 'Partial From' 'Partial To'};
    file_sheets = validate_xls(temp,'T');

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
