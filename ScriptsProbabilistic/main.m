% [INPUT]
% src = The file containing the dataset.
% des = The name of the file to which the results are written.
%
% [NOTE]
% The financial time series in the dataset must have been previously validated:
%  - illiquid series with too many zeroes have been discarded;
%  - rows with NaNs have been removed or filled with interpolation;
%  - there are enough observations to run consistent calculations;
%  - etc...

function main(src,des)

    if (exist(src,'file') == 0)
        error('The source file does not exist.');
    end

    k = 0.05;
    l = 0.08;

    firms = get_firms_count(src);
    rm = get_market_index(src);
    sv = get_state_vars(src);

    res = cell(firms,1);

    for i = 1:firms     
        dx = get_firm_liabilities(src,i);
        ex = get_firm_capitalization(src,i);
        rx = get_firm_returns(src,i);

        r = [rm rx];
        r = r - (ones(length(r),1) * mean(r));

        [~,~,~,p,~,~,~,~,~,~,~,s] = dcc_gjrgarch(r,1,1,1,1);
        sm = sqrt(s(:,1));
        sx = sqrt(s(:,2));
        pmx = squeeze(p(1,2,:));
        
        rdm = r(:,1);
        rdx = r(:,2);
        
        betax = pmx .* (sx ./ sm);
        varx = sx * quantile((rdx ./ sx),k);

        if (isempty(sv))
            [~,dcovar] = calculate_covar(rdm,rdx,varx,k);
        else
            [~,dcovar] = calculate_covar(rdm,rdx,varx,k,sv);
        end
        
        [mes,lrmes] = calculate_mes(rdm,sm,rdx,sx,pmx,k);
        srisk = calculate_srisk(lrmes,dx,ex,l);

        res{i} = [betax (varx .* -1) dcovar mes srisk];
    end
    
    if (exist(des,'file') == 2)
        delete(des);
    end
    
    write_results(des,res);

end
