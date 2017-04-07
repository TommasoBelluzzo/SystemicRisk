% [INPUT]
% file_src = A string representing the name of the Excel spreadsheet containing the dataset (optional, default='dataset.xlsx').
% file_des = A string representing the name of the Excel spreadsheet to which the results are written, eventually replacing the previous ones (optional, default='results.xlsx').
% k        = A float representing the confidence level used to calculate various measures (optional, default=0.95).
% l        = A float representing the capital adequacy ratio used to calculate SRISK (optional, default=0.08).
% anl      = A boolean that indicates whether to analyse the results (optional, default=false).
%
% [NOTES]
% This function produces no outputs, its purpose is to save the results into an Excel spreadsheet and, optionally, analyse them.

function main(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addOptional('file_src','dataset.xlsx',@(x)validateattributes(x,{'char'},{'nonempty','size',[1,NaN]}));
        ip.addOptional('file_des','results.xlsx',@(x)validateattributes(x,{'char'},{'nonempty','size',[1,NaN]}));
        ip.addOptional('k',0.95,@(x)validateattributes(x,{'double','single'},{'scalar','real','finite','>=',0.90,'<=',0.99}));
        ip.addOptional('l',0.08,@(x)validateattributes(x,{'double','single'},{'scalar','real','finite','>=',0.05,'<=',0.20}));
        ip.addOptional('anl',false,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});
    ip_res = ip.Results;
    
    [path,~,~] = fileparts(pwd);
    [~,name,ext] = fileparts(ip_res.file_src);
    file_src = fullfile(path,[name ext]);

    [~,name,ext] = fileparts(ip_res.file_des);
    
    if (~strcmp(ext,'.xlsx'))
        file_des = fullfile(pwd,[name ext '.xlsx']);
    else
        file_des = fullfile(pwd,[name ext]); 
    end
    
    main_internal(file_src,file_des,ip_res.k,ip_res.l,ip_res.anl);

end

function main_internal(file_src,file_des,k,l,anl)

    addpath('../ScriptsCommon');
    data = parse_dataset(file_src);
    rmpath('../ScriptsCommon');

    data.A = 1 - k;
    data.K = k;
    data.L = l;
    data.Beta = NaN(data.Obs,data.Frms);
    data.VaR = NaN(data.Obs,data.Frms);
    data.CoVaR = NaN(data.Obs,data.Frms);
    data.DCoVaR = NaN(data.Obs,data.Frms);
    data.MES = NaN(data.Obs,data.Frms);
    data.SRISK = NaN(data.Obs,data.Frms);
   
    ret0_m = data.IdxRet - mean(data.IdxRet);
    
    bar = waitbar(0,'Calculating firms measures...','CreateCancelBtn','setappdata(gcbf,''stop'',true)');
    setappdata(bar,'stop',false);
    
    try
        for i = 1:data.Frms
            waitbar(((i - 1) / data.Frms),bar,sprintf('Calculating measures for %s and %s...',data.IdxNam,data.FrmsNam{i}));

            if (getappdata(bar,'stop'))
                delete(bar);
                return;
            end

            ret_x = data.FrmsRet(:,i);
            ret0_x = ret_x - mean(ret_x);

            [p,s] = dcc_gjrgarch([ret0_m ret0_x]);
            s_m = sqrt(s(:,1));
            s_x = sqrt(s(:,2));
            p_mx = squeeze(p(1,2,:));

            beta = p_mx .* (s_x ./ s_m);
            
            [var,covar,dcovar] = calculate_covar(ret0_m,ret0_x,s_x,data.A,data.StVarsLag);
            [mes,lrmes] = calculate_mes(ret0_m,s_m,ret0_x,s_x,p_mx,data.A);
            srisk = calculate_srisk(lrmes,data.FrmsLia(:,i),data.FrmsCap(:,i),l);

            data.Beta(:,i) = beta;
            data.VaR(:,i) = var;
            data.CoVaR(:,i) = covar;
            data.DCoVaR(:,i) = dcovar;
            data.MES(:,i) = mes;
            data.SRISK(:,i) = srisk;

            if (getappdata(bar,'stop'))
                delete(bar);
                return;
            end
            
            waitbar((i / data.Frms),bar);
        end

        mcaps_sum = sum(data.FrmsCap,2);
        mcaps_wei = data.FrmsCapLag ./ repmat(sum(data.FrmsCapLag,2),1,data.Frms);
        beta_avg = sum(data.Beta .* mcaps_wei,2) .* mcaps_sum;
        var_avg = sum(data.VaR .* mcaps_wei,2) .* mcaps_sum;
        covar_avg = sum(data.CoVaR .* mcaps_wei,2) .* mcaps_sum;
        dcovar_avg = sum(data.DCoVaR .* mcaps_wei,2) .* mcaps_sum;
        mes_avg = sum(data.MES .* mcaps_wei,2) .* mcaps_sum;
        srisk_avg = sum(data.SRISK .* mcaps_wei,2);
        data.Avgs = [beta_avg var_avg covar_avg dcovar_avg mes_avg srisk_avg];

        waitbar(100,bar,'Writing results...');
        write_results(file_des,data);
        
        delete(bar);
        
        if (anl)        
            data.DatesLim = [data.DatesNum(1) data.DatesNum(end)];

            plot_index(data);
            plot_averages(data);
            plot_correlations(data);
        end
    catch e
        delete(bar);
        rethrow(e);
    end

end

function write_results(file_des,data)

    if (exist(file_des,'file') == 2)
        delete(file_des);
    end

    dates_str = cell2table(data.DatesStr,'VariableNames',{'Date'});
    covar = [dates_str array2table(data.CoVaR,'VariableNames',data.FrmsNam)];
    dcovar = [dates_str array2table(data.DCoVaR,'VariableNames',data.FrmsNam)];
    mes = [dates_str array2table(data.MES,'VariableNames',data.FrmsNam)];
    srisk = [dates_str array2table(data.SRISK,'VariableNames',data.FrmsNam)];
    avgs = [dates_str array2table(data.Avgs(:,3:end),'VariableNames',{'CoVaR' 'DCoVaR' 'MES' 'SRISK'})];
    
    writetable(covar,file_des,'FileType','spreadsheet','Sheet',1,'WriteRowNames',true);
    writetable(dcovar,file_des,'FileType','spreadsheet','Sheet',2,'WriteRowNames',true);
    writetable(mes,file_des,'FileType','spreadsheet','Sheet',3,'WriteRowNames',true);
    writetable(srisk,file_des,'FileType','spreadsheet','Sheet',4,'WriteRowNames',true);    
    writetable(avgs,file_des,'FileType','spreadsheet','Sheet',5,'WriteRowNames',true);    

    exc = actxserver('Excel.Application');
    exc_wbs = exc.Workbooks.Open(file_des,0,false);
    exc_wbs.Sheets.Item(1).Name = 'CoVaR';
    exc_wbs.Sheets.Item(2).Name = 'DCoVaR';
    exc_wbs.Sheets.Item(3).Name = 'MES';
    exc_wbs.Sheets.Item(4).Name = 'SRISK';
    exc_wbs.Sheets.Item(5).Name = 'Averages';   
    exc_wbs.Save();
    exc_wbs.Close();
    exc.Quit();

end

function plot_index(data)

    tit = ['Market Index (' data.IdxNam ')'];

    fig = figure();
    set(fig,'Name',tit,'Units','normalized','Position',[100 100 0.6 0.6]);

    sub_1 = subplot(2,1,1);
    plot(sub_1,data.DatesNum,data.IdxRet,'-b');
    datetick(sub_1,'x','yyyy','KeepLimits');
    xlabel(sub_1,'Time');
    ylabel(sub_1,'Returns');
    set(sub_1,'XLim',data.DatesLim,'YLim',[(min(data.IdxRet) - 0.01) (max(data.IdxRet) + 0.01)]);
    title(sub_1,'Log Returns');
    
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

    suptitle(tit);
    movegui(fig,'center');

end

function plot_averages(data)

    avgs = data.Avgs(:,3:end) ./ 1e6;
    x_max = max(max(avgs));
    x_min = min(min(avgs));
    y_lim = [(x_min - (x_min * 0.1)) (x_max - (x_max * 0.1))];
    
    fig = figure();
    set(fig,'Name','Averages','Units','normalized','Position',[100 100 0.6 0.6]);

    sub_1 = subplot(2,2,1);
    plot(sub_1,data.DatesNum,avgs(:,1));
    datetick(sub_1,'x','yyyy');
    xlabel(sub_1,'Time');
    ylabel(sub_1,'Billions of Dollars');
    title(sub_1,'CoVaR');
    
    sub_2 = subplot(2,2,2);
    plot(sub_2,data.DatesNum,avgs(:,2));
    datetick(sub_2,'x','yyyy','KeepLimits');
    xlabel(sub_2,'Time');
    ylabel(sub_2,'Billions of Dollars');
    title(sub_2,'Delta CoVaR');

    sub_3 = subplot(2,2,3);
    plot(sub_3,data.DatesNum,avgs(:,3));
    datetick(sub_3,'x','yyyy','KeepLimits');
    xlabel(sub_3,'Time');
    ylabel(sub_3,'Billions of Dollars');
    title(sub_3,'MES');
    
    sub_4 = subplot(2,2,4);
    plot(sub_4,data.DatesNum,avgs(:,4));
    datetick(sub_4,'x','yyyy','KeepLimits');
    xlabel(sub_4,'Time');
    ylabel(sub_4,'Billions of Dollars');
    title(sub_4,'SRISK');

    set([sub_1 sub_2 sub_3 sub_4],'XLim',data.DatesLim,'YLim',y_lim);

    y_lbls = arrayfun(@(x) sprintf('%.0f%',x),(get(gca,'YTick') .* 100),'UniformOutput',false);
    set([sub_1 sub_2 sub_3 sub_4],'YTickLabel',y_lbls);
    
    suptitle('Averages');
    movegui(fig,'center');

end

function plot_correlations(data)

    meas = {'Beta' 'VaR' 'CoVaR' 'DCoVaR' 'MES' 'SRISK'};

    [rho,pval] = corr(data.Avgs);
    m = mean(data.Avgs);
    s = std(data.Avgs);
    z = bsxfun(@minus,data.Avgs,m);
    z = bsxfun(@rdivide,z,s);
    z_lims = [nanmin(z(:)) nanmax(z(:))];

    fig = figure();
    set(fig,'Name','Correlation Matrix','Units','normalized','Position',[100 100 0.6 0.6]);

    [h,axes,big_ax] = gplotmatrix(data.Avgs,[],[],[],'o',2,[],'hist',meas,meas);
    set(h(logical(eye(6))),'FaceColor',[0.678 0.922 1]);

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
    movegui(fig,'center');

end
