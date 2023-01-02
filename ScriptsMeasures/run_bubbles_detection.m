% [INPUT]
% ds = A structure representing the dataset.
% sn = A string representing the serial number of the result file.
% temp = A string representing the full path to the Excel spreadsheet used as template for the result file.
% out = A string representing the full path to the Excel spreadsheet to which the results are written, eventually replacing the previous ones.
% cvm = A string representing the method for calculating the critical values (optional, default='WB'):
%   - 'FS' for finite sample critical values;
%   - 'WB' for wild bootstrap.
% cvq = A float [0.90,0.99] representing the quantile of the critical values (optional, default=0.95).
% lag_max = An integer [0,10] representing the maximum lag order to be evaluated for the Augmented Dickey-Fuller test (optional, default=0).
% lag_sel = A string representing the lag order selection criteria for the Augmented Dickey-Fuller test (optional, default='FIX'):
%   - 'AIC' for Akaike's Information Criterion;
%   - 'BIC' for Bayesian Information Criterion;
%   - 'FIX' to use a fixed lag order;
%   - 'FPE' for Final Prediction Error;
%   - 'HQIC' for Hannan-Quinn Information Criterion.
% mbd = An integer [3,Inf) representing the minimum duration of a bubble in days (optional, default=NaN).
%   If NaN is provided, then an optimal value based on the number of observations is used.
% analyze = A boolean that indicates whether to analyse the results and display plots (optional, default=false).
%
% [OUTPUT]
% result = A structure representing the original dataset inclusive of intermediate and final calculations.
% stopped = A boolean that indicates whether the process has been stopped through user input.

function [result,stopped] = run_bubbles_detection(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('ds',@(x)validateattributes(x,{'struct'},{'nonempty'}));
        ip.addRequired('sn',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addRequired('temp',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addRequired('out',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addOptional('cvm','WB',@(x)any(validatestring(x,{'FS' 'WB'})));
        ip.addOptional('cvq',0.95,@(x)validateattributes(x,{'double'},{'real' 'finite' '>=' 0.90 '<=' 0.99 'scalar'}));
        ip.addOptional('lag_max',0,@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 0 '<=' 10 'scalar'}));
        ip.addOptional('lag_sel','FIX',@(x)any(validatestring(x,{'AIC' 'BIC' 'FIX' 'FPE' 'HQIC'})));
        ip.addOptional('mbd',NaN,@(x)validateattributes(x,{'double'},{'real' 'scalar'}));
        ip.addOptional('analyze',false,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    ds = validate_dataset(ipr.ds,'BubblesDetection');
    sn = ipr.sn;
    temp = validate_template(ipr.temp);
    out = validate_output(ipr.out);
    cvm = ipr.cvm;
    cvq = ipr.cvq;
    lag_max = ipr.lag_max;
    lag_sel = ipr.lag_sel;
    mbd = validate_mbd(ipr.mbd,ds.T);
    analyze = ipr.analyze;

    nargoutchk(1,2);

    [result,stopped] = run_bubbles_detection_internal(ds,sn,temp,out,cvm,cvq,lag_max,lag_sel,mbd,analyze);

end

function [result,stopped] = run_bubbles_detection_internal(ds,sn,temp,out,cvm,cvq,lag_max,lag_sel,mbd,analyze)

    result = [];
    stopped = false;
    e = [];

    ds = initialize(ds,sn,cvm,cvq,lag_max,lag_sel,mbd);
    n = ds.N;
    t = ds.T;

    bar = waitbar(0,'Initializing bubbles detection measures...','CreateCancelBtn',@(src,event)setappdata(gcbf(),'Stop', true));
    setappdata(bar,'Stop',false);
    cleanup = onCleanup(@()delete(bar));

    pause(1);
    waitbar(0,bar,'Calculating bubbles detection measures...');
    pause(1);

    try

        p = ds.Prices;

        for i = 1:n
            waitbar((i - 1) / n,bar,['Calculating bubbles detection measures for ' ds.FirmNames{i} '...']);

            if (getappdata(bar,'Stop'))
                stopped = true;
                break;
            end

            
            offset = min([(ds.Defaults(i) - 1) (ds.Insolvencies(i) - 1) t]);
            p_i = p(1:offset,i);

            [bsadfs,cvs,detection,breakdown] = psy_bubbles_detection(p_i,ds.CVM,ds.CVQ,ds.LagMax,ds.LagSel,ds.MBD);
            ds.BSADF(1:offset,i) = bsadfs;
            ds.CVS(1:offset,i) = cvs;
            ds.BUB(1:offset,i) = detection(:,1);
            ds.BMPH(1:offset,i) = detection(:,2);
            ds.BRPH(1:offset,i) = detection(:,3);
            ds.Breakdowns{i} = breakdown;

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
    waitbar(1,bar,'Writing bubbles detection measures...');
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

function ds = initialize(ds,sn,cvm,cvq,lag_max,lag_sel,mbd)

    n = ds.N;
    t = ds.T;

    ds.Result = 'BubblesDetection';
    ds.ResultDate = now(); %#ok<TNOW1> 
    ds.ResultAnalysis = @(ds)analyze_result(ds);
    ds.ResultSerial = sn;

    ds.CVM = cvm;
    ds.CVQ = cvq;
    ds.LagMax = lag_max;
    ds.LagSel = lag_sel;
    ds.MBD = mbd;

    if (strcmp(ds.LagSel,'FIX'))
        label = [' (CVM= ' ds.CVM ', CVQ=' num2str(ds.CVQ * 100) '%, L=' num2str(ds.LagMax) ')'];
    else
        label = [' (CVM= ' ds.CVM ', CVQ=' num2str(ds.CVQ * 100) '%, LM=' num2str(ds.LagMax) ', LS=' ds.LagSel ')'];
    end

    ds.LabelsMeasuresSimple = {'BUB' 'BMPH' 'BRPH' 'Breakdowns'};
    ds.LabelsMeasures = [strcat(ds.LabelsMeasuresSimple(1:3),{label}) ds.LabelsMeasuresSimple(4)];

    ds.LabelsIndicatorsSimple = {'BC' 'BCP'};
    ds.LabelsIndicators = {['BC' label] ['BCP' label]};

    ds.LabelsSheetsSimple = [ds.LabelsMeasuresSimple {'Indicators'}];
    ds.LabelsSheets = [ds.LabelsMeasures {'Indicators'}];

    ds.BSADF = NaN(t,n);
    ds.CVS = NaN(t,n);
    ds.BUB = NaN(t,n);
    ds.BMPH = NaN(t,n);
    ds.BRPH = NaN(t,n);
    ds.Breakdowns = cell(1,n);

    ds.Indicators = NaN(t,numel(ds.LabelsIndicators));

    ds.ComparisonReferences = {'Indicators' 1 strcat({'BD-'},ds.LabelsIndicatorsSimple{1})};

end

function ds = finalize(ds)

    bc = sum(ds.Bubbles .* ds.Capitalizations,2,'omitnan');
    bcp = bc ./ sum(ds.Capitalizations,2,'omitnan');

    ds.Indicators = [bc bcp];

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

    lc = numel(ds.LabelsSheetsSimple);

    dates_str = cell2table(ds.DatesStr,'VariableNames',{'Date'});

    for i = 1:(lc - 2)
        sheet = ds.LabelsSheetsSimple{i};
        measure = strrep(sheet,' ','');

        tab = [dates_str array2table(ds.(measure),'VariableNames',ds.FirmNames)];
        writetable(tab,out,'FileType','spreadsheet','Sheet',sheet,'WriteRowNames',true);
    end

    breakdowns_count = cellfun(@(x)size(x,1),ds.Breakdowns);
    breakdowns_firms = arrayfun(@(x)repelem(ds.FirmNames(x),breakdowns_count(x),1),1:ds.N,'UniformOutput',false);
    breakdowns_firms = vertcat(breakdowns_firms{:});
    breakdowns_firms = cell2table(breakdowns_firms,'VariableNames',{'Firm'});

    tab = [breakdowns_firms array2table(cell2mat(ds.Breakdowns.'),'VariableNames',{'Start' 'Peak' 'End' 'Duration', 'Boom Phase' 'Burst Phase'})];
    writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{lc - 1},'WriteRowNames',true);

    tab = [dates_str array2table(ds.Indicators,'VariableNames',ds.LabelsIndicatorsSimple)];
    writetable(tab,out,'FileType','spreadsheet','Sheet',ds.LabelsSheetsSimple{lc},'WriteRowNames',true);

    worksheets_batch(out,ds.LabelsSheetsSimple,ds.LabelsSheets);

end

%% PLOTTING

function analyze_result(ds)


end

%% VALIDATION

function mbd = validate_mbd(mbd,t)

    if (~isnan(mbd))
        if (~isfinite(mbd))
            error('The value of ''mbd'' is invalid. Expected input to be finite.');
        end
    
        if (floor(mbd) ~= mbd)
            error('The value of ''mbd'' is invalid. Expected input to be an integer.');
        end

        b = ceil(0.2 * t);

        if ((mbd < 3) || (mbd > b))
            error(['The value of ''mbd'' is invalid. Expected input to have a value >= 5 and <= ' num2str(b) '.']);
        end
    end

end

function out = validate_output(out)

    [path,name,extension] = fileparts(out);

    if (~strcmpi(extension,'.xlsx'))
        out = fullfile(path,[name extension '.xlsx']);
    end

end

function temp = validate_template(temp)

    sheets = {'BUB' 'BMPH' 'BRPH' 'Breakdowns' 'Indicators'};
    file_sheets = validate_xls(temp,'T');

    if (~all(ismember(sheets,file_sheets)))
        error(['The template must contain the following sheets: ' sheets{1} sprintf(', %s',sheets{2:end}) '.']);
    end

    worksheets_batch(temp,sheets);

end
