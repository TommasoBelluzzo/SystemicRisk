% [INPUT]
% file_src = The file containing the dataset.
% file_des = The name of the file to which the results are written.
%
% [NOTE]
% The financial time series in the dataset must have been previously validated and preprocessed:
%  - there are enough observations to run consistent calculations;
%  - illiquid series with too many zeroes have been discarded;
%  - outliers have been detected and removed;
%  - rows with NaNs have been removed or filled with interpolation;
%  - etc...

function main(file_src,file_des)

    if (exist(file_src,'file') == 0)
        error('The source file does not exist.');
    end

    k = 0.05;
    l = 0.08;

    n = get_firms_count(file_src);
    svars = get_state_variables(file_src);

    ret_m = get_market_index_returns(file_src);
    rdem_m = ret_m - mean(ret_m);
    
    res = cell(n,1);

    for i = 1:n     
        tl_x = get_firm_total_liabilities(file_src,i);
        mc_x = get_firm_market_capitalization(file_src,i);
        ret_x = get_firm_returns(file_src,i);
        rdem_x = ret_x - mean(ret_x);

        [~,~,~,r,~,~,~,~,~,~,~,s2] = dcc_gjrgarch([rdem_m rdem_x],1,1,1,1);
        s_m = sqrt(s2(:,1));
        s_x = sqrt(s2(:,2));
        rho = squeeze(r(1,2,:));
        
        b_x = rho .* (s_x ./ s_m);
        var_x = s_x * quantile((rdem_x ./ s_x),k);

        if (isempty(svars))
            [~,dcovar] = calculate_covar(rdem_m,rdem_x,var_x,k);
        else
            [~,dcovar] = calculate_covar(rdem_m,rdem_x,var_x,k,svars);
        end
        
        [mes,lrmes] = calculate_mes(rdem_m,s_m,rdem_x,s_x,rho,k);
        srisk = calculate_srisk(lrmes,tl_x,mc_x,l);

        res{i} = [b_x (var_x .* -1) dcovar mes srisk];
    end
    
    if (exist(file_des,'file') == 2)
        delete(file_des);
    end
    
    write_results(file_des,res);

end
