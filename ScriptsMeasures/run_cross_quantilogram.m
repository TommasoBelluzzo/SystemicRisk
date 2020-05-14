% [INPUT]
% ds = A structure representing the dataset.
% temp = A string representing the full path to the Excel spreadsheet used as a template for the results file.
% out = A string representing the full path to the Excel spreadsheet to which the results are written, eventually replacing the previous ones.
% bw = An integer [21,252] representing the dimension of each rolling window (optional, default=252).
% a = A float [0.01,0.10] representing the target quantile (optional, default=0.05).
% lags = An integer [10,60] representing the maximum number of lags (optional, default=60).
% ci_m = A string (either 'SB' for Stationary Bootstrap or 'SN' for Self-normalization) representing the computational approach of confidence intervals (optional, default='SB').
% ci_s = A float {0.005;0.010;0.025;0.050;0.100} representing the significance level of confidence intervals (optional, default=0.050).
% ci_p = Optional argument whose type depends on the the chosen computational approach of confidence intervals:
%   - for Stationary Bootstrap, an integer [10,1000] representing the number of bootstrap iterations (default=100);
%   - for Self-normalization, a float {0.00;0.01;0.03;0.05;0.10;0.15;0.20;0.30} representing the fraction of the sample that corresponds to the minimum subsample size (default=0.10).
% analyze = A boolean that indicates whether to analyse the results and display plots (optional, default=false).
%
% [OUTPUT]
% result = A structure representing the original dataset inclusive of intermediate and final calculations.
% stopped = A boolean that indicates whether the process has been stopped through user input.
%
% [NOTES]
% Credit goes to Kevin Sheppard, the author of the original code.

function [result,stopped] = run_cross_quantilogram(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('ds',@(x)validateattributes(x,{'struct'},{'nonempty'}));
        ip.addRequired('temp',@(x)validateattributes(x,{'char'},{'nonempty','size',[1,NaN]}));
        ip.addRequired('out',@(x)validateattributes(x,{'char'},{'nonempty','size',[1,NaN]}));
        ip.addOptional('bw',252,@(x)validateattributes(x,{'double'},{'real','finite','integer','>=',21,'<=',252,'scalar'}));
        ip.addOptional('a',0.05,@(x)validateattributes(x,{'double'},{'real','finite','>=',0.01,'<=',0.10,'scalar'}));
        ip.addOptional('lags',60,@(x)validateattributes(x,{'double'},{'real','finite','integer','>=',10,'<=',60,'scalar'}));
        ip.addOptional('ci_m','SB',@(x)any(validatestring(x,{'SB','SN'})));
        ip.addOptional('ci_s',0.050,@(x)validateattributes(x,{'double'},{'real','finite','>',0,'<=',0.1,'scalar'}));
        ip.addOptional('ci_p',NaN,@(x)validateattributes(x,{'double'},{'real','finite','scalar'}));
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
    [ci_m,ci_s,ci_p,ci_v] = validate_ci_input(ipr.ci_m,ipr.ci_s,ipr.ci_p);
    analyze = ipr.analyze;
    
    nargoutchk(1,2);

    [result,stopped] = run_cross_quantilogram_internal(ds,temp,out,bw,a,lags,ci_m,ci_s,ci_p,ci_v,analyze);

end

function [result,stopped] = run_cross_quantilogram_internal(ds,temp,out,bw,a,lags,ci_m,ci_s,ci_p,ci_v,analyze)

    result = [];
    stopped = false;
    e = [];

    ds = initialize(ds,bw,a,lags,ci_m,ci_s,ci_p,ci_v);
    n = ds.N;
    t = ds.T;

    step_1 = 0.2;
    step_2 = 1 - step_1;
    
    rng(double(bitxor(uint16('T'),uint16('B'))));
	cleanup_1 = onCleanup(@()rng('default'));

    bar = waitbar(0,'Initializing cross-quantilogram measures...','CreateCancelBtn',@(src,event)setappdata(gcbf(),'Stop', true));
    setappdata(bar,'Stop',false);
    cleanup_2 = onCleanup(@()delete(bar));
    
    pause(1);
    waitbar(0,bar,'Calculating cross-quantilogram measures (step 1 of 2)...');
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

                futures(i) = parfeval(@main_loop_1,1,idx_i,r_i,sv_i,ds.A,ds.Lags,ds.CIM,ds.CIS,ds.CIP,ds.CIV);
            end
        else
            for i = 1:n
                offset = min(ds.Defaults(i) - 1,t);
                idx_i = idx(1:offset);
                r_i = r(1:offset,i);

                futures(i) = parfeval(@main_loop_1,1,idx_i,r_i,[],ds.A,ds.Lags,ds.CIM,ds.CIS,ds.CIP,ds.CIV);
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
    waitbar(step_1,bar,'Finalizing cross-quantilogram measures (step 1 of 2)...');
    pause(1);

    try
        ds = finalize_1(ds,futures_results);
    catch e
        delete(bar);
        rethrow(e);
    end
    
    pause(1);
    waitbar(step_1,bar,'Calculating cross-quantilogram measures (step 2 of 2)...');
    pause(1);
    
    try

        windows_idx = extract_rolling_windows(ds.Index,ds.BW);
        windows_r = extract_rolling_windows(ds.Returns,ds.BW);
        
        if (ds.PI)
            windows_sv = extract_rolling_windows(ds.StateVariables,ds.BW);
        else
            windows_sv = repmat({[]},ds.T,1);
        end

        futures(1:t) = parallel.FevalFuture;
        futures_max = 0;
        futures_results = cell(t,1);

        for i = 1:t
            futures(i) = parfeval(@main_loop_2,1,windows_idx{i},windows_r{i},windows_sv{i},ds.A,ds.Lags,ds.CIM,ds.CIS,ds.CIP,ds.CIV);
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
    waitbar(1,bar,'Finalizing cross-quantilogram measures (step 2 of 2)...');
    pause(1);

    try
        ds = finalize_2(ds,futures_results);
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
        safe_plot(@(id)plot_windows(ds,'Full',id));
        
        if (ds.PI)
            safe_plot(@(id)plot_sequence(ds,'Partial',id));
            safe_plot(@(id)plot_windows(ds,'Partial',id));
        end
    end
    
    result = ds;

end

%% DATA

function ds = initialize(ds,bw,a,lags,ci_m,ci_s,ci_p,ci_v)

    n = ds.N;
    t = ds.T;

    ds.A = a;
    ds.BW = bw;
    ds.CIM = ci_m;
    ds.CIP = ci_p;
    ds.CIS = ci_s;
    ds.CIV = ci_v;
    ds.Lags = lags;
    ds.PI = ~isempty(ds.StateVariables);

    ds.CQFullFrom = NaN(lags,n,3);
    ds.CQFullTo = NaN(lags,n,3);
    ds.CQFullFromWindows = NaN(t,3);
    ds.CQFullToWindows = NaN(t,3);
    
    if (ds.PI)
        ds.CQPartialFrom = NaN(lags,n,3);
        ds.CQPartialTo = NaN(lags,n,3);
        ds.CQPartialFromWindows = NaN(t,3);
        ds.CQPartialToWindows = NaN(t,3);
    end

end

function ds = finalize_1(ds,window_results)
  
    n = ds.N;

    for i = 1:n
        window_result = window_results{i};
        
        for j = 1:3
            ds.CQFullFrom(:,i,j) = window_result.CQFull(:,1,j);
            ds.CQFullTo(:,i,j) = window_result.CQFull(:,2,j);
        end
    end
    
    if (ds.PI)
        for i = 1:n
            window_result = window_results{i};

            for j = 1:3
                ds.CQPartialFrom(:,i,j) = window_result.CQPartial(:,1,j);
                ds.CQPartialTo(:,i,j) = window_result.CQPartial(:,2,j);
            end
        end
    end

end

function ds = finalize_2(ds,window_results)
  
    t = ds.T;
    alpha = 2 / (ds.BW + 1);

    for i = 1:t
        window_result = window_results{i};
        ds.CQFullFromWindows(i,:) = window_result.CQFull(1,:);
        ds.CQFullToWindows(i,:) = window_result.CQFull(2,:);
    end

    for i = 1:3
        from = ds.CQFullFromWindows(:,i);
        ds.CQFullFromWindows(:,i) = [from(1); filter(alpha,[1 (alpha - 1)],from(2:end),(1 - alpha) * from(1))];
        
        to = ds.CQFullToWindows(:,i);
        ds.CQFullToWindows(:,i) = [to(1); filter(alpha,[1 (alpha - 1)],to(2:end),(1 - alpha) * to(1))];
    end
    
    if (ds.PI)
        for i = 1:t
            window_result = window_results{i};
            ds.CQPartialFromWindows(i,:) = window_result.CQPartial(1,:);
            ds.CQPartialToWindows(i,:) = window_result.CQPartial(2,:);
        end
        
        for i = 1:3
            from = ds.CQPartialFromWindows(:,i);
            ds.CQPartialFromWindows(:,i) = [from(1); filter(alpha,[1 (alpha - 1)],from(2:end),(1 - alpha) * from(1))];
            
            to = ds.CQPartialToWindows(:,i);
            ds.CQPartialToWindows(:,i) = [to(1); filter(alpha,[1 (alpha - 1)],to(2:end),(1 - alpha) * to(1))];
        end
    end

end

function [ci_m,ci_s,ci_p,ci_v] = validate_ci_input(ci_m,ci_s,ci_p)

    ci_s_allowed = [0.005 0.010 0.025 0.050 0.100];
	[ci_s_ok,j] = ismember(ci_s,ci_s_allowed);

    if (~ci_s_ok)
        ci_s_allowed_text = [sprintf('%.2f',ci_s_allowed(1)) sprintf(', %.2f',ci_s_allowed(2:end))];
        error(['The CI significance must have one of the following values: ' ci_s_allowed_text]);
    end

    if (strcmp(ci_m,'SB'))
        if (isnan(ci_p))
            ci_p = 100;
        else
            validateattributes(ci_p,{'double'},{'scalar','integer','real','finite','>=',10,'<=',1000});
        end
        
        ci_v = [];
    else
        ci_p_allowed = [0.00 0.01 0.03 0.05 0.10 0.15 0.20 0.30];
        
        if (isnan(ci_p))
            ci_p = 0.10;
        end
        
        [ci_p_ok,i] = ismember(ci_p,ci_p_allowed);
        
        if (~ci_p_ok)
            ci_p_allowed_text = [sprintf('%.2f',ci_p_allowed(1)) sprintf(', %.2f',ci_p_allowed(2:end))];
            error(['The CI parameter must have one of the following values: ' ci_p_allowed_text]);
        end

        cv = [
            129.15490  99.44085  66.00439 45.43917 28.06313;
            131.82880 101.21590  66.49058 45.48538 28.31850;
            131.83560 101.31700  67.55465 46.02739 28.51908;
            135.24000 103.32090  68.20319 46.48723 28.82658;
            139.34750 106.80970  71.07829 48.55132 30.01695;
            150.60740 115.90190  76.13708 51.38898 31.66900;
            166.53440 127.71000  83.45021 55.85094 34.03793;
            206.01210 155.00120 101.10960 67.53906 40.84930
        ];
        
        ci_v = cv(i,j);
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

    sheets = {'Full From' 'Full To' 'Full From Windows' 'Full To Windows' 'Partial From' 'Partial To' 'Partial From Windows' 'Partial To Windows'};
    
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
    
    lags = num2cell(repelem(1:ds.Lags,1,3).');
    labels = repmat({'CQ' 'Lower CI' 'Upper CI'},1,ds.Lags).';
    row_headers = [labels lags];

	dates_str = cell2table(ds.DatesStr,'VariableNames',{'Date'});

    vars = [row_headers num2cell(reshape(permute(ds.CQFullFrom,[3 1 2]),ds.Lags * 3,ds.N))];
    tab = cell2table(vars,'VariableNames',[{'Value' 'Lag'} ds.FirmNames]);
    writetable(tab,out,'FileType','spreadsheet','Sheet','Full From','WriteRowNames',true);
    
    vars = [row_headers num2cell(reshape(permute(ds.CQFullTo,[3 1 2]),ds.Lags * 3,ds.N))];
    tab = cell2table(vars,'VariableNames',[{'Value' 'Lag'} ds.FirmNames]);
    writetable(tab,out,'FileType','spreadsheet','Sheet','Full To','WriteRowNames',true);
    
    tab = [dates_str array2table(ds.CQFullFromWindows,'VariableNames',{'CQ' 'Lower_CI' 'Upper_CI'})];
    writetable(tab,out,'FileType','spreadsheet','Sheet','Full From Windows','WriteRowNames',true);
    
    tab = [dates_str array2table(ds.CQFullToWindows,'VariableNames',{'CQ' 'Lower_CI' 'Upper_CI'})];
    writetable(tab,out,'FileType','spreadsheet','Sheet','Full To Windows','WriteRowNames',true);
    
    if (ds.PI)
        vars = [row_headers num2cell(reshape(permute(ds.CQPartialFrom,[3 1 2]),ds.Lags * 3,ds.N))];
        tab = cell2table(vars,'VariableNames',[{'Value' 'Lag'} ds.FirmNames]);
        writetable(tab,out,'FileType','spreadsheet','Sheet','Partial From','WriteRowNames',true);

        vars = [row_headers num2cell(reshape(permute(ds.CQPartialTo,[3 1 2]),ds.Lags * 3,ds.N))];
        tab = cell2table(vars,'VariableNames',[{'Value' 'Lag'} ds.FirmNames]);
        writetable(tab,out,'FileType','spreadsheet','Sheet','Partial To','WriteRowNames',true);

        tab = [dates_str array2table(ds.CQPartialFromWindows,'VariableNames',{'CQ' 'Lower_CI' 'Upper_CI'})];
        writetable(tab,out,'FileType','spreadsheet','Sheet','Partial From Windows','WriteRowNames',true);

        tab = [dates_str array2table(ds.CQPartialToWindows,'VariableNames',{'CQ' 'Lower_CI' 'Upper_CI'})];
        writetable(tab,out,'FileType','spreadsheet','Sheet','Partial To Windows','WriteRowNames',true);
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
                exc_wb.Sheets.Item('Partial From Windows').Delete();
                exc_wb.Sheets.Item('Partial To Windows').Delete();

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

end

%% MEASURES

function window_results = main_loop_1(idx,r,sv,a,lags,ci_m,ci_s,ci_p,ci_v)

    window_results = struct();

    from = [r idx];
    to = [idx r];
    cq_full = zeros(lags,2,3);

    if (strcmp(ci_m,'SB'))
        for k = 1:lags
            [cq,cv] = calculate_stationary_bootstrap(from,a,k,ci_s,ci_p);
            cq_full(k,1,:) = [cq cv];

            [cq,cv] = calculate_stationary_bootstrap(to,a,k,ci_s,ci_p);
            cq_full(k,2,:) = [cq cv];
        end
    else
        for k = 1:lags
            [cq,cv] = calculate_self_normalization(from,a,k,ci_v,ci_p);
            cq_full(k,1,:) = [cq cv];

            [cq,cv] = calculate_self_normalization(to,a,k,ci_v,ci_p);
            cq_full(k,2,:) = [cq cv];
        end
    end
    
    window_results.CQFull = cq_full;
    
    if (~isempty(sv))
        from = [r idx sv];
        to = [idx r sv];
        cq_partial = zeros(lags,2,3);

        if (strcmp(ci_m,'SB'))
            for k = 1:lags
                [cq,cv] = calculate_stationary_bootstrap(from,a,k,ci_s,ci_p);
                cq_partial(k,1,:) = [cq cv];

                [cq,cv] = calculate_stationary_bootstrap(to,a,k,ci_s,ci_p);
                cq_partial(k,2,:) = [cq cv];
            end
        else
            for k = 1:lags
                [cq,cv] = calculate_self_normalization(from,a,k,ci_v,ci_p);
                cq_partial(k,1,:) = [cq cv];

                [cq,cv] = calculate_self_normalization(to,a,k,ci_v,ci_p);
                cq_partial(k,2,:) = [cq cv];
            end
        end
        
        window_results.CQPartial = cq_partial;
    end

end

function window_results = main_loop_2(idx,r,sv,a,lag,ci_m,ci_s,ci_p,ci_v)

    window_results = struct();

	nan_indices = sum(isnan(r),1) > 0;
    r(:,nan_indices) = [];
    
    [t,n] = size(r);

    sv_empty = isempty(sv);
    
    if (t < (lag * 2))
        d = (lag * 2) - t;

        idx = [idx; normrnd(mean(idx),std(idx),[d 1])];

        mu = ones(d,1) .* mean(r,1);
        sigma = ones(d,1) .* std(r,1);
		
        rho = corr(r);
        rho(isnan(rho)) = 0;
		
		if (any(eig(rho) <= 0))
		    rho = rho + (1e-16 .* eye(n))
        end

        r = [r; (normrnd(mu,sigma,[d n]) * chol(rho,'upper'))];

        if (~sv_empty)
            mu = ones(d,1) .* mean(sv,1);
            sigma = ones(d,1) .* std(sv,1);
            sv = [sv; normrnd(mu,sigma,[d size(sv,2)])];
        end
    end
    
    cq_full = zeros(n,2,3);

    for i = 1:n
        r_i = r(:,i);
        
        from = [r_i idx];
        to = [idx r_i];

        if (strcmp(ci_m,'SB'))
            [cq,cv] = calculate_stationary_bootstrap(from,a,lag,ci_s,ci_p);
            cq_full(i,1,:) = [cq cv];

            [cq,cv] = calculate_stationary_bootstrap(to,a,lag,ci_s,ci_p);
            cq_full(i,2,:) = [cq cv];
        else
            [cq,cv] = calculate_self_normalization(from,a,lag,ci_v,ci_p);
            cq_full(i,1,:) = [cq cv];

            [cq,cv] = calculate_self_normalization(to,a,lag,ci_v,ci_p);
            cq_full(i,2,:) = [cq cv];
        end
    end
    
    window_results.CQFull = squeeze(mean(cq_full,1));
    
    if (~sv_empty)
        cq_partial = zeros(n,2,3);
        
        for i = 1:n
            r_i = r(:,i);
        
            from = [r_i idx sv];
            to = [idx r_i sv];

            if (strcmp(ci_m,'SB'))
                [cq,cv] = calculate_stationary_bootstrap(from,a,1,ci_s,ci_p);
                cq_partial(i,1,:) = [cq cv];

                [cq,cv] = calculate_stationary_bootstrap(to,a,1,ci_s,ci_p);
                cq_partial(i,2,:) = [cq cv];
            else
                [cq,cv] = calculate_self_normalization(from,a,1,ci_v,ci_p);
                cq_partial(i,1,:) = [cq cv];

                [cq,cv] = calculate_self_normalization(to,a,1,ci_v,ci_p);
                cq_partial(i,2,:) = [cq cv];
            end
        end
        
        window_results.CQPartial = squeeze(mean(cq_partial,1));
    end

end

function [cq,cv] = calculate_self_normalization(x,a,k,v,f)

    [t,n] = size(x);
    len = max(round(f * t,0),1);
    partial = n > 2;

    cq_sn = zeros(t,1);
    
    if (partial)
        for i = len:t
            x_i = x(1:i,:);
            x_t = size(x_i,1);

            q_sn = (x_i <= repmat(gumbel_quantile(x_i,a),x_t,1)) - (ones(x_t,n) .* a);

            d_sn = zeros(x_t-k,n);
            d_sn(:,1) = q_sn(k+1:x_t,1);
            d_sn(:,2:n) = q_sn(1:x_t-k,2:n);

            h_sn = d_sn.' * d_sn;
            
            if (det(h_sn) <= 1e-08)
                hi_sn = pinv(h_sn);
            else
                hi_sn = inv(h_sn);
            end
            
            cq_sn(i) = -hi_sn(1,2) / sqrt(hi_sn(1,1) * hi_sn(2,2));
        end 
    else
        for i = len:t
            x_i = x(1:i,:);
            x_t = size(x_i,1);

            q_sn = (x_i <= repmat(gumbel_quantile(x_i,a),x_t,1)) - (ones(x_t,n) .* a);

            d_sn = zeros(x_t-k,n);
            d_sn(:,1) = q_sn(k+1:x_t,1);
            d_sn(:,2:n) = q_sn(1:x_t-k,2:n);

            h_sn = d_sn.' * d_sn;

            cq_sn(i) = h_sn(1,2) / sqrt(h_sn(1,1) * h_sn(2,2));
        end
    end

    cq = cq_sn(end);
    
    cqc = (cq_sn - repmat(cq,t,1)) .* (1:t).';
    cv0_bb = cqc(len:t);
    cv0_sn = (t * cq^2) / ((1 / t^2) * (cv0_bb.' * cv0_bb));
    cv0 = sqrt(v * (cq^2 / cv0_sn));
    cv = [-cv0 cv0];

end

function [cq,cv] = calculate_stationary_bootstrap(x,a,k,s,b)

    [t,n] = size(x);
    len = t - k;
    partial = n > 2;
    
    s = s / 2;
    
    d = zeros(len,n);
    d(:,1) = x(k+1:t,1);
    d(:,2:n) = x(1:len,2:n);
    
    block_length = ppw_optimal_block_length(d);
    g = mean(block_length(:,1));
    
    a_sb = ones(len,n) .* a;
    cq_sb = zeros(b,1);

    if (partial)
        for i = 1:b
            indices = indices_bootstrap(len,g);

            d_sb = d(indices,:);
            q_sb = (d_sb <= repmat(gumbel_quantile(d_sb,a),len,1)) - a_sb;
            
            h_sb = q_sb.' * q_sb;
            
            if (det(h_sb) <= 1e-08)
                hi_sb = pinv(h_sb);
            else
                hi_sb = inv(h_sb);
            end

            cq_sb(i) = -hi_sb(1,2) / sqrt(hi_sb(1,1) * hi_sb(2,2));
        end
    else
        for i = 1:b
            indices = indices_bootstrap(len,g);

            d_sb = d(indices,:);
            q_sb = (d_sb <= repmat(gumbel_quantile(d_sb,a),len,1)) - a_sb;
            
            h_sb = q_sb.' * q_sb;

            cq_sb(i) = h_sb(1,2) / sqrt(h_sb(1,1) * h_sb(2,2));
        end
    end
    
    q = (x <= repmat(gumbel_quantile(x,a),t,1)) - (ones(t,n) .* a);
    
    d = zeros(len,n);
    d(:,1) = q(k+1:t,1);
    d(:,2:n) = q(1:len,2:n);
    
    h = d.' * d;

    if (partial)
        if (det(h) <= 1e-08)
            hi = pinv(h);
        else
            hi = inv(h);
        end

        cq = -hi(1,2) / sqrt(hi(1,1) * hi(2,2));
    else
        cq = h(1,2) / sqrt(h(1,1) * h(2,2));
    end

    cqc = cq_sb - cq;
    cv = [min(0,gumbel_quantile(cqc,s)) max(0,gumbel_quantile(cqc,1 - s))];

end

function q = gumbel_quantile(x,p)

    index = 1 + ((size(x,1) - 1) * p);
    low = floor(index);
    high = ceil(index);
    
    x = sort(x);
    x_low = x(low,:);
    x_high = x(high,:);
    
    h = max(index - low,0);
    q = (h .* x_high) + ((1 - h) .* x_low);

end

function indices = indices_bootstrap(n,g)

    indices = [ceil(n * rand()); zeros(n - 1,1)];

    u = rand(n,1) < g;
    indices(u) = ceil(n .* rand(sum(u),1));

    zi = find(~u(2:n));
    indices(zi + 1) = indices(zi) + 1;

    fi = indices > n;
    indices(fi) = indices(fi) - n;

end

function bl = ppw_optimal_block_length(x)

    [t,n] = size(x);

    k = max(sqrt(log10(t)),5);
    c = 2 * sqrt(log10(t) / t);

    b_max = ceil(min(3 * sqrt(t),t / 3));
    m_max = ceil(sqrt(t)) + k;

    bl = zeros(n,2);

    for ppw_i = 1:n
        x_i = x(:,ppw_i);

        p1 = m_lag(x_i,m_max);
        p1 = p1(m_max+1:end,:);
        p1 = corr([x_i(m_max+1:end) p1]);
        p1 = p1(2:end,1);

        p2 = [m_lag(p1,k).' p1(end-k+1:end)];
        p2 = p2(:,k+1:end);
        p2 = sum((abs(p2) < (ones(k,m_max - k + 1) .* c))).';

        p3 = [(1:length(p2)).' p2];
        p3 = p3(p2 == k,:);

        if (isempty(p3))
            m_hat = find(abs(p1) > c,1,'last');
        else
            m_hat = p3(1,1);
        end

        m = min(2 * m_hat,m_max);

        if (m > 0)
            mm = (-m:m).';

            p1 = m_lag(x_i,m);
            p1 = p1(m+1:end,:);
            p1 = cov([x_i(m+1:end),p1]);

            act = sortrows([-(1:m).' p1(2:end,1)],1);
            ac = [act(:,2); p1(:,1)];

            mmn = mm ./ m;
            kernel_weights = ((abs(mmn) >= 0) .* (abs(mmn) < 0.5)) + (2 .* (1 - abs(mmn)) .* (abs(mmn) >= 0.5) .* (abs(mmn) <= 1));

            acw = kernel_weights .* ac;
            acw_ss = sum(acw)^2;

            g_hat = sum(acw .* abs(mm));
            dcb_hat = (4/3) * acw_ss;
            dsb_hat = 2 * acw_ss;

            b_comp1 = 2 * g_hat^2;
            b_comp2 = t^(1 / 3);
            bl_vl = min((b_comp1 / dsb_hat)^(1/3) * b_comp2,b_max);
            bl_cb = min((b_comp1 / dcb_hat)^(1/3) * b_comp2,b_max);

            bl(ppw_i,:) = [bl_vl bl_cb];
        else
            bl(ppw_i,:) = 1;
        end
    end
    
    function l = m_lag(x,n)

        mn = numel(x);
        l = ones(mn,n);

        for ml_i = 1:n
            l(ml_i+1:mn,ml_i) = x(1:mn-ml_i,1);
        end

    end

end

%% PLOTTING

function plot_sequence(ds,target,id)

    lags = ds.Lags;

    x = (1:lags).';
    x_limits = [0.3 (lags + 0.7)];
    
    if (lags <= 20)
        x_tick = 1:lags;
    else
        x_tick = [1 10:10:lags];
    end

    if (ds.PI)
        y_limits = plot_limits([ds.CQFullFrom(:) ds.CQFullTo(:) ds.CQPartialFrom(:) ds.CQPartialTo(:)],0.1);
    else
        y_limits = plot_limits([ds.CQFullFrom(:) ds.CQFullTo(:)],0.1);
    end

    core = struct();

    core.N = ds.N;
    core.PlotFunction = @(subs,x,y)plot_function(subs,x,y);
    core.SequenceFunction = @(y,offset)reshape(y(:,offset,:),lags,6);
	
    core.OuterTitle = 'Cross-Quantilogram Measures';
    core.InnerTitle = [target ' Cross-Quantilograms (Lags)'];

    core.Plots = 2;
	core.PlotsLabels = ds.FirmNames;
    core.PlotsTitle = {'From' 'To'};
    core.PlotsType = 'V';

    core.X = x;
    core.XDates = [];
    core.XGrid = false;
    core.XLabel = [];
    core.XLimits = x_limits;
    core.XRotation = [];
    core.XTick = x_tick;
    core.XTickLabels = [];

    core.Y = cat(3,ds.(['CQ' target 'From']),ds.(['CQ' target 'To']));
    core.YGrid = false;
    core.YLabel = [];
    core.YLimits = y_limits;
    core.YRotation = [];
    core.YTick = [];
    core.YTickLabels = @(x)sprintf('%.2f',x);

    sequential_plot(core,id);
    
    function plot_function(subs,x,y)

        for i = 0:1
            sub = subs(i+1);
            
            j = i * 3;

            bar(sub,x,y(:,j+1),0.6,'EdgeColor','none','FaceColor',[0.749 0.862 0.933]);
            hold(sub,'on');
                plot(sub,x,y(:,j+2),'Color',[1 0.4 0.4],'LineWidth',1);
                plot(sub,x,y(:,j+3),'Color',[1 0.4 0.4],'LineWidth',1);
            hold(sub,'off');
        end
        
    end

end

function plot_windows(ds,target,id)

    from = ds.(['CQ' target 'FromWindows']);
    [from_cq,from_cil,from_cih] = deal(from(:,1),from(:,2),from(:,3));
    
    below_indices = (from_cq < 0) & (from_cq < from_cil);
    from_cq_below = NaN(ds.T,1);
    from_cq_below(below_indices) = from_cq(below_indices);
    
    above_indices = (from_cq > 0) & (from_cq > from_cih);
    from_cq_above = NaN(ds.T,1);
    from_cq_above(above_indices) = from_cq(above_indices);
    
    to = ds.(['CQ' target 'ToWindows']);
    [to_cq,to_cil,to_cih] = deal(to(:,1),to(:,2),to(:,3));
    
    below_indices = (to_cq < 0) & (to_cq < to_cil);
    to_cq_below = NaN(ds.T,1);
    to_cq_below(below_indices) = to_cq(below_indices);
    
    above_indices = (to_cq > 0) & (to_cq > to_cih);
    to_cq_above = NaN(ds.T,1);
    to_cq_above(above_indices) = to_cq(above_indices);

    text = [target ' Cross-Quantilograms (Windows EWMA)'];

    y_limits = plot_limits([from_cq to_cq],0.1);

    f = figure('Name',['Cross-Quantilogram Measures > ' text],'Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    sub_1 = subplot(1,2,1);
    line(sub_1,[ds.DatesNum(1) ds.DatesNum(end)],[0 0],'Color',[0 0 0]);
    hold on;
        plot(sub_1,ds.DatesNum,from_cq,'Color',[0.000 0.447 0.741]);
        plot(sub_1,ds.DatesNum,from_cq_below,'Color',[1.000 0.400 0.400]);
        plot(sub_1,ds.DatesNum,from_cq_above,'Color',[1.000 0.400 0.400]);
    hold off;
    t1 = title(sub_1,'From');
    set(t1,'Units','normalized');
    t1_position = get(t1,'Position');
    set(t1,'Position',[0.4783 t1_position(2) t1_position(3)]);

    sub_2 = subplot(1,2,2);
    line(sub_2,[ds.DatesNum(1) ds.DatesNum(end)],[0 0],'Color',[0 0 0]);
    hold on;
        plot(sub_2,ds.DatesNum,to_cq,'Color',[0.000 0.447 0.741]);
        plot(sub_2,ds.DatesNum,to_cq_below,'Color',[1.000 0.400 0.400]);
        plot(sub_2,ds.DatesNum,to_cq_above,'Color',[1.000 0.400 0.400]);
    hold off;
    t2 = title(sub_2,'To');
    set(t2,'Units','normalized');
    t2_position = get(t2,'Position');
    set(t2,'Position',[0.4783 t2_position(2) t2_position(3)]);
    
    set([sub_1 sub_2],'XLim',[ds.DatesNum(1) ds.DatesNum(end)],'XTickLabelRotation',45);
    set([sub_1 sub_2],'YLim',y_limits);
    set([sub_1 sub_2],'XGrid','on','YGrid','on');

    if (ds.MonthlyTicks)
        date_ticks([sub_1 sub_2],'x','mm/yyyy','KeepLimits','KeepTicks');
    else
        date_ticks([sub_1 sub_2],'x','yyyy','KeepLimits');
    end
    
    y_tick = get(sub_1,'YTick');
    y_labels = arrayfun(@(x)sprintf('%.2f',x),y_tick,'UniformOutput',false);
    set([sub_1 sub_2],'YTick',y_tick,'YTickLabel',y_labels);

    figure_title(text);
    
    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end
