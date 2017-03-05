% [INPUT]
% source      = The file containing the necessary data.
% destination = The name of the file to which the results are written.

function main(source,destination)  

    if (exist(source, 'file') == 0)
        error('The source file does not exist.');
    end

    firms = get_firms_count(source);
    k = 0.05; l = 0.08;
    rm = get_market_index(source);
    sv = get_state_vars(source);

    results = cell(firms,1);

    for i = 1:firms     
        dx = get_firm_liabilities(source,i);
        ex = get_firm_capitalization(source,i);
        rx = get_firm_returns(source,i);

        data = [dx ex rm rx sv];
        data(any(isnan(data),2),:) = [];

        r = [data(:,3) data(:,4)];
        r = r - (ones(length(r),1) * mean(r));
        rdm = r(:,1);
        rdx = r(:,2);
        
        [~,p,~,~,~,s] = dcc_gjr_garch([rdm rdx],1,1,1,1);
        sm = sqrt(s(:,1));
        sx = sqrt(s(:,2));
        pmx = squeeze(p(1,2,:));
        
        bx = pmx .* (sx ./ sm);
        varx = sx * quantile((rdx ./ sx),k);

        if (isempty(sv))
            [~,dcovar] = calculate_covar(rdm,rdx,varx,k);
        else
            [~,dcovar] = calculate_covar(rdm,rdx,varx,k,data(:,5:end));
        end
        
        [mes,lrmes] = calculate_mes(rdm,sm,rdx,sx,pmx,k);

        dx = data(:,1);
        ex = data(:,2);
        srisk = calculate_srisk(l,lrmes,dx,ex);

        results{i} = [bx (varx .* -1) dcovar mes srisk];
    end
    
    if (exist(destination, 'file') == 2)
        delete(destination);
    end
    
    write_results(destination,results);

end