% [INPUT]
% data = A structure representing the dataset.
% res  = A string representing the full path to the Excel spreadsheet to which the results are written, eventually replacing the previous ones.
% k    = A float representing the confidence level used to calculate various measures (optional, default=0.95).
% d    = A float representing the six-month crisis threshold for the market index decline used to calculate LRMES (optional, default=0.40).
% l    = A float representing the capital adequacy ratio used to calculate SRISK (optional, default=0.08).
% anl  = A boolean that indicates whether to analyse the results (optional, default=false).
%
% [NOTES]
% This function produces no outputs, its purpose is to save the results into an Excel spreadsheet and, optionally, analyse them.

function main_pro(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data',@(x)validateattributes(x,{'struct'},{'nonempty'}));
        ip.addRequired('res',@(x)validateattributes(x,{'char'},{'nonempty','size',[1,NaN]}));
        ip.addOptional('k',0.95,@(x)validateattributes(x,{'double','single'},{'scalar','real','finite','>=',0.90,'<=',0.99}));
        ip.addOptional('d',0.40,@(x)validateattributes(x,{'double','single'},{'scalar','real','finite','>=',0.05,'<=',0.99}));
        ip.addOptional('l',0.08,@(x)validateattributes(x,{'double','single'},{'scalar','real','finite','>=',0.05,'<=',0.20}));
        ip.addOptional('anl',false,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});

    ip_res = ip.Results;
    res = ip_res.res;

    [path,name,ext] = fileparts(res);

    if (~strcmp(ext,'.xlsx'))
        res = fullfile(path,[name ext '.xlsx']);
    end
    
    main_pro_internal(ip_res.data,res,ip_res.k,ip_res.d,ip_res.l,ip_res.anl);

end

function main_pro_internal(data,res,k,d,l,anl)

    data = initialize_data(data,k,d,l);

    ret0_m = data.IdxRet - mean(data.IdxRet);
    
    bar = waitbar(0,'Calculating probabilistc measures...','CreateCancelBtn','setappdata(gcbf,''Stop'',true)');
    setappdata(bar,'Stop',false);
    
    try
        for i = 1:data.Frms
            waitbar(((i - 1) / data.Frms),bar,sprintf('Calculating probabilistc measures for %s and %s...',data.IdxNam,data.FrmsNam{i}));

            if (getappdata(bar,'Stop'))
                delete(bar);
                return;
            end

            ret_x = data.FrmsRet(:,i);
            ret0_x = ret_x - mean(ret_x);

            [p,s] = dcc_gjrgarch([ret0_m ret0_x]);
            s_m = sqrt(s(:,1));
            s_x = sqrt(s(:,2));
            p_mx = squeeze(p(1,2,:));

            beta_x = p_mx .* (s_x ./ s_m);
            var_x = s_x * quantile((ret0_x ./ s_x),data.A);

            [covar,dcovar] = calculate_covar(ret0_m,ret0_x,var_x,data.A,data.StVarsLag);
            [mes,lrmes] = calculate_mes(ret0_m,s_m,ret0_x,s_x,beta_x,p_mx,data.A,data.D);
            srisk = calculate_srisk(lrmes,data.FrmsLia(:,i),data.FrmsCap(:,i),data.L);

            data.Beta(:,i) = beta_x;
            data.VaR(:,i) = -1 .* var_x;
            data.CoVaR(:,i) = -1 .* covar;
            data.DCoVaR(:,i) = -1 .* dcovar;
            data.MES(:,i) = -1 .* mes;
            data.SRISK(:,i) = srisk;

            if (getappdata(bar,'Stop'))
                delete(bar);
                return;
            end
            
            waitbar((i / data.Frms),bar);
        end

        data = finalize_data(data);
        
        waitbar(100,bar,'Writing probabilistc measures...');
        write_results(res,data);
        
        delete(bar);
        
        if (anl)
            plot_index(data);
            plot_averages(data);
            plot_correlations(data);
        end
    catch e
        delete(bar);
        rethrow(e);
    end

end

function data = initialize_data(data,k,d,l)
  
    data.A = 1 - k;
    data.D = d;
    data.K = k;
    data.L = l;
    data.Beta = NaN(data.Obs,data.Frms);
    data.VaR = NaN(data.Obs,data.Frms);
    data.CoVaR = NaN(data.Obs,data.Frms);
    data.DCoVaR = NaN(data.Obs,data.Frms);
    data.MES = NaN(data.Obs,data.Frms);
    data.SRISK = NaN(data.Obs,data.Frms);

end

function data = finalize_data(data)

    k_lbl = sprintf('%.0f%%',(data.K * 100));
    d_lbl = sprintf('%.0f%%',(data.D * 100));
    l_lbl = sprintf('%.0f%%',(data.L * 100));
    
    data.Lbls = {'Beta' ['VaR (k=' k_lbl ')'] ['CoVaR (k=' k_lbl ')'] ['DCoVaR (k=' k_lbl ')'] ['MES (k=' k_lbl ')'] ['SRISK (d=' d_lbl ' l=' l_lbl ')'] 'Averages'};
    data.LblsSim = {'Beta' 'VaR' 'CoVaR' 'DCoVaR' 'MES' 'SRISK' 'Averages'};   

    mcaps_sum = sum(data.FrmsCap,2);
    wei = data.FrmsCapLag ./ repmat(sum(data.FrmsCapLag,2),1,data.Frms);
    beta_avg = sum(data.Beta .* wei,2) .* mcaps_sum;
    var_avg = sum(data.VaR .* wei,2) .* mcaps_sum;
    covar_avg = sum(data.CoVaR .* wei,2) .* mcaps_sum;
    dcovar_avg = sum(data.DCoVaR .* wei,2) .* mcaps_sum;
    mes_avg = sum(data.MES .* wei,2) .* mcaps_sum;
    srisk_avg = sum(data.SRISK .* wei,2);

    data.Avgs = [beta_avg var_avg covar_avg dcovar_avg mes_avg srisk_avg];

end

function write_results(res,data)

    [res_path,~,~] = fileparts(res);

    if (exist(res_path,'dir') ~= 7)
        mkdir(res_path);
    end

    if (exist(res,'file') == 2)
        delete(res);
    end

    dates_str = cell2table(data.DatesStr,'VariableNames',{'Date'});
    covar = [dates_str array2table(data.CoVaR,'VariableNames',data.FrmsNam)];
    dcovar = [dates_str array2table(data.DCoVaR,'VariableNames',data.FrmsNam)];
    mes = [dates_str array2table(data.MES,'VariableNames',data.FrmsNam)];
    srisk = [dates_str array2table(data.SRISK,'VariableNames',data.FrmsNam)];
    avgs = [dates_str array2table(data.Avgs(:,3:end),'VariableNames',data.LblsSim(3:end-1))];

    writetable(covar,res,'FileType','spreadsheet','Sheet',1,'WriteRowNames',true);
    writetable(dcovar,res,'FileType','spreadsheet','Sheet',2,'WriteRowNames',true);
    writetable(mes,res,'FileType','spreadsheet','Sheet',3,'WriteRowNames',true);
    writetable(srisk,res,'FileType','spreadsheet','Sheet',4,'WriteRowNames',true);    
    writetable(avgs,res,'FileType','spreadsheet','Sheet',5,'WriteRowNames',true);    

    exc = actxserver('Excel.Application');
    exc_wbs = exc.Workbooks.Open(res,0,false);
    
    for i = 1:5
        exc_wbs.Sheets.Item(i).Name = data.Lbls{i+2};
    end
  
    exc_wbs.Save();
    exc_wbs.Close();
    exc.Quit();

end

function plot_index(data)

    fig = figure('Name',['Market Index (' data.IdxNam ')'],'Units','normalized','Position',[100 100 0.85 0.85]);

    sub_1 = subplot(2,1,1);
    plot(sub_1,data.DatesNum,data.IdxRet,'-b');
    datetick(sub_1,'x','yyyy','KeepLimits');
    xlabel(sub_1,'Time');
    ylabel(sub_1,'Returns');
    set(sub_1,'XLim',[data.DatesNum(1) data.DatesNum(end)],'YLim',[(min(data.IdxRet) - 0.01) (max(data.IdxRet) + 0.01)]);
    t1 = title(sub_1,'Log Returns');
    set(t1,'Units','normalized');
    t1_pos = get(t1,'Position');
    set(t1,'Position',[0.4783 t1_pos(2) t1_pos(3)]);
    
    sub_2 = subplot(2,1,2);
    hist = histogram(sub_2,data.IdxRet,50,'FaceAlpha',0.25,'Normalization','pdf');
    hold on;
        edg = get(hist,'BinEdges');
        edg_max = max(edg);
        edg_min = min(edg);
        [f,x] = ksdensity(data.IdxRet);
        plot(sub_2,x,f,'-b','LineWidth',1.5);
    hold off;
    strs = {sprintf('Observations: %d',size(data.IdxRet,1)) sprintf('Kurtosis: %.4f',kurtosis(data.IdxRet)) sprintf('Mean: %.4f',mean(data.IdxRet)) sprintf('Median: %.4f',median(data.IdxRet)) sprintf('Skewness: %.4f',skewness(data.IdxRet)) sprintf('Standard Deviation: %.4f',std(data.IdxRet))};
    annotation('TextBox',(get(sub_2,'Position') - [0 0.03 0 0]),'String',strs,'EdgeColor','none','FitBoxToText','on','FontSize',8);
    set(sub_2,'XLim',[(edg_min - (edg_min * 0.1)) (edg_max - (edg_max * 0.1))]);
    title(sub_2,'P&L Distribution');
    t2 = title(sub_2,'P&L Distribution');
    set(t2,'Units','normalized');
    t2_pos = get(t2,'Position');
    set(t2,'Position',[0.4783 t2_pos(2) t2_pos(3)]);

    t = figure_title(['Market Index (' data.IdxNam ')']);
    t_pos = get(t,'Position');
    set(t,'Position',[t_pos(1) -0.0157 t_pos(3)]);
    
    pause(0.01);

    jfr = get(fig,'JavaFrame');
    set(jfr,'Maximized',true);

end

function plot_averages(data)

    avgs = data.Avgs(:,3:end) ./ 1e6;
    avgs_len = size(avgs,2);

    x_max = max(max(avgs));
    x_max_sign = sign(x_max);
    x_min = min(min(avgs));
    x_min_sign = sign(x_min);

    y_lim = [((abs(x_min) * 1.1) * x_min_sign) ((abs(x_max) * 1.1) * x_max_sign)];

    fig = figure('Name','Averages','Units','normalized','Position',[100 100 0.85 0.85]);

    subs = NaN(avgs_len,1);
    
    for i = 1:avgs_len
        sub = subplot(2,2,i);
        plot(sub,data.DatesNum,avgs(:,i));
        datetick(sub,'x','yyyy');
        xlabel(sub,'Time');
        ylabel(sub,'Billions of Dollars');
        title(sub,data.Lbls(i+2));
        
        subs(i) = sub;
    end

    set(subs,'XLim',[data.DatesNum(1) data.DatesNum(end)],'YLim',y_lim);

    y_lbls = arrayfun(@(x)sprintf('%.0f',x),(get(gca,'YTick') .* 100),'UniformOutput',false);
    set(subs,'YTickLabel',y_lbls);
    
    t = figure_title('Averages');
    t_pos = get(t,'Position');
    set(t,'Position',[t_pos(1) -0.0157 t_pos(3)]);

    pause(0.01);

    jfr = get(fig,'JavaFrame');
    set(jfr,'Maximized',true);

end

function plot_correlations(data)

    meas = data.LblsSim(1:end-1);
    
    [rho,pval] = corr(data.Avgs);
    m = mean(data.Avgs);
    s = std(data.Avgs);
    z = bsxfun(@minus,data.Avgs,m);
    z = bsxfun(@rdivide,z,s);
    z_lims = [nanmin(z(:)) nanmax(z(:))];

    fig = figure('Name','Correlation Matrix','Units','normalized');
    
    pause(0.01);
    jfr = get(fig,'JavaFrame');
    set(jfr,'Maximized',true);

    pause(0.01);
    set(0,'CurrentFigure',fig);
    [h,axes,big_ax] = gplotmatrix(data.Avgs,[],[],[],'o',2,[],'hist',meas,meas);
    set(h(logical(eye(6))),'FaceColor',[0.678 0.922 1]);
    
    drawnow();

    x_lbls = get(axes,'XLabel');
    y_lbls = get(axes,'YLabel');
    set([x_lbls{:}; y_lbls{:}],'FontWeight','bold');

    lim_ij = 1:6;
    
    for i = lim_ij
        for j = lim_ij
            ax_ij = axes(i,j);
            
            z_lims_cur = 1.1 .* z_lims;
            x_lim = m(j) + (z_lims_cur * s(j));
            y_lim = m(i) + (z_lims_cur * s(i));
            
            set(get(big_ax,'Parent'),'CurrentAxes',ax_ij);
            set(ax_ij,'XLim',x_lim,'XTick',[],'YLim',y_lim,'YTick',[]);
            axis normal;
            
            if (i ~= j)
                hls = lsline();
                set(hls,'Color','r');

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

end
