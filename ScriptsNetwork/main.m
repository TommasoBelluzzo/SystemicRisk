% [INPUT]
% file_src = The file containing the dataset.
% file_des = The name of the file to which the results are written.
%
% [NOTE]
% The financial time series in the dataset must have been previously validated:
%  - illiquid series with too many zeroes have been discarded;
%  - rows with NaNs have been removed or filled with interpolation;
%  - there are enough observations to run consistent calculations;
%  - etc...

function main(file_src,file_des)

    if (exist(file_src,'file') == 0)
        error('The source file does not exist.');
    end

    grp = [5 11 18];
    rob = 1;
    sst = 0.05;

    ret = get_table_slice(read_sheet(file_src,1),1,0,3,0);
    %ret = get_firms_returns(file_src);
    win = get_rolling_windows(ret,262);
    win_len = length(win);
    
    res = cell(win_len,1);
    
    for i = 1:win_len
        ret_i = win{i,1};
        
        adjm = calculate_adjacency_matrix(ret_i,sst,rob);

        [dci,num_io,num_ioo,clo_cen,deg_cen,eig_cen,clust] = calculate_measures(adjm,grp);

        res{i}.dci = dci;
        res{i}.num_io = num_io;
        res{i}.num_ioo = num_ioo;
        res{i}.clo_cen = clo_cen;
        res{i}.deg_cen = deg_cen;   
        res{i}.eig_cen = eig_cen; 
        res{i}.clust = clust;         
    end

    if (exist(file_des,'file') == 2)
        delete(file_des);
    end

    write_results(file_des,res);

end