% [INPUT]
% file_src = A string representing the name of the Excel spreadsheet containing the dataset (optional, default='dataset.xlsx').
% file_des = A string representing the name of the Excel spreadsheet to which the results are written, eventually replacing the previous ones (optional, default='results.xlsx').
% sst      = A float representing he statistical significance threshold for the linear Granger-causality test (optional, default=0.05).
% rob      = A boolean indicating whether to use robust p-values (optional, default=true).
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
        ip.addOptional('sst',0.05,@(x)validateattributes(x,{'double','single'},{'scalar','real','finite','>',0,'<=',0.20}));
        ip.addOptional('rob',true,@(x)validateattributes(x,{'logical'},{'scalar'}));
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
    
    main_internal(file_src,file_des,ip_res.sst,ip_res.rob,ip_res.anl);

end

function main_internal(file_src,file_des,sst,rob,anl)

    addpath('../ScriptsCommon');
    data = parse_dataset(file_src);
    rmpath('../ScriptsCommon');

    data.SST = sst;
    data.Rob = rob;

    win = get_rolling_windows(data.RetFrms,252);
    win_len = length(win);
    
    res = cell(win_len,1);
    
    for i = 1:win_len
        ret_i = win{i,1};
        [~,~,~,~,pcae,~] = pca(ret_i);

        adjm = calculate_adjacency_matrix(ret_i,sst,rob);
        [dci,n_io,n_ioo,cloc,cluc,degc,eigc] = calculate_measures(adjm,grp);

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
