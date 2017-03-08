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

    grp = [5; 11; 18];
    rob = 1;
    sst = 0.05;

    ret = get_firms_returns(file_src);
    win = get_rolling_windows(ret,262);
    win_len = length(win);
    
    res = cell(win_len,1);
    
    for i = 1:win_len
        ret_i = win{i,1};
        [~,~,~,~,pcae,~] = pca(ret_i);

        adjm = calculate_adjacency_matrix(ret_i,sst,rob);
        [dci,n_io,n_ioo,cloc,degc,eigc,cluc] = calculate_measures(adjm,grp);

        res{i}.dci = dci;
        res{i}.num_io = n_io;
        res{i}.num_ioo = n_ioo;
        res{i}.cloc = cloc;
        res{i}.cluc = cluc;
        res{i}.degc = degc;
        res{i}.eigc = eigc;
        res{i}.pcae = pcae;
    end

    if (exist(file_des,'file') == 2)
        delete(file_des);
    end

    write_results(file_des,res);

end
