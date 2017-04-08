% [INPUT]
% data = A numeric t-by-n matrix containing the network data.
% sst  = A float representing he statistical significance threshold for the linear Granger-causality test (optional, default=0.05).
% rob  = A boolean indicating whether to use robust p-values (optional, default=true).
%
% [OUTPUT]
% adjm = An numeric n-by-n matrix representing the adjacency matrix of the network.

function adjm = calculate_adjacency_matrix(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data',@(x)validateattributes(x,{'numeric'},{'2d','finite','nonempty','nonnan','real'}));
        ip.addOptional('sst',0.05,@(x)validateattributes(x,{'double','single'},{'scalar','real','finite','>',0,'<=',0.20}));
        ip.addOptional('rob',true,@(x)validateattributes(x,{'logical'},{'scalar'}));
    end

    ip.parse(varargin{:});
    ip_res = ip.Results;
    
    adjm = calculate_adjacency_matrix_internal(ip_res.data,ip_res.sst,ip_res.rob);

end

function adjm = calculate_adjacency_matrix_internal(data,sst,rob)

    n = size(data,2);
    adjm = zeros(n);
    ij_seq = 1:n;

    for i = ij_seq
        data_in = data(:,i);

        parfor j = ij_seq
            if (i == j)
                continue;
            end
            
            data_out = data(:,j);
            
            if (rob)
                [pval,~] = linear_granger_causality(data_in, data_out);
            else
                [~,pval] = linear_granger_causality(data_in, data_out);
            end

            if (pval < sst)
                adjm(i,j) = 1;
            end
        end
    end

end