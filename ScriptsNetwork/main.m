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
    data = update_data(data,sst,rob);

    win = get_rolling_windows(data.FrmsRet,252);
    win_len = length(win);
    win_dif = data.Obs - win_len;

    bar = waitbar(0,'Calculating measures...','CreateCancelBtn','setappdata(gcbf,''stop'',true)');
    setappdata(bar,'stop',false);
    
    try
        for i = 1:win_len
            waitbar(((i - 1) / win_len),bar,sprintf('Calculating measures for window %d of %d...',i,win_len));
            
            if (getappdata(bar,'stop'))
                delete(bar);
                return;
            end
            
            win_i = win{i,1};
            win_off = i + win_dif;
            
            [~,~,~,~,pcae,~] = pca(win_i);
            [pca_coe,~,pca_exp] = pcacov(corrcoef(win_i));

            adjm = calculate_adjacency_matrix(win_i,data.SST,data.Rob);
            [dci,n_io,n_ioo,cloc,cluc,degc,eigc] = calculate_measures(adjm,data.GrpsSep);

            data.AdjMats{win_off} = adjm;
            data.CloC(win_off,:) = cloc;
            data.CluC(win_off,:) = cluc;
            data.DCI(win_off) = dci;
            data.DegC(win_off,:) = degc;
            data.EigC(win_off,:) = eigc;
            data.NumIO(win_off) = n_io;
            data.NumIOO(win_off) = n_ioo;
            data.PCACoe{win_off} = pca_coe;
            data.PCAExp{win_off} = pca_exp;

            if (getappdata(bar,'stop'))
                delete(bar);
                return;
            end
            
            waitbar((i / win_len),bar,sprintf('Calculating measures for window %d of %d...',i,win_len));
        end
        
        waitbar(100,bar,'Writing results...');
        write_results(file_des,data);
        
        delete(bar);
        
        if (anl)        

        end
    catch e
        delete(bar);
        rethrow(e);
    end

end

function data = update_data(data,sst,rob)

    data.SST = sst;
    data.Rob = rob;
    
	data.AdjMats = cell(data.Obs,1);
    data.CloC = NaN(data.Obs,data.Frms);
    data.CluC = NaN(data.Obs,data.Frms);
	data.DCI = NaN(data.Obs,1);
    data.DegC = NaN(data.Obs,data.Frms);
    data.EigC = NaN(data.Obs,data.Frms);
	data.NumIO = NaN(data.Obs,1);
	data.NumIOO = NaN(data.Obs,1);
	data.PCACoe = cell(data.Obs,1);
	data.PCAExp = cell(data.Obs,1);

end