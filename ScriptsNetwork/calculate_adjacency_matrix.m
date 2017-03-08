% [INPUT]
% ret  = A t-by-n matrix containing firms log returns.
% sst  = The statistical significance threshold for the linear Granger-causality Test.
% rob  = A boolean indicating whether to use standard or robust p-values.
%
% [OUTPUT]
% adjm = An n-by-n matrix representing the adjacency matrix of the network.

function adjm = calculate_adjacency_matrix(ret,sst,rob)

    n = size(ret,2);
    adjm = zeros(n);

    for i = 1:n
        ret_in = ret(:,i);

        parfor j = 1:n
            if (i == j)
                continue;
            end
            
            ret_out = ret(:,j);
            
            if (rob == 0)
                [pval,~] = linear_granger_causality(ret_in, ret_out);
            else
                [~,pval] = linear_granger_causality(ret_in, ret_out);
            end

            if (pval < sst)
                adjm(i,j) = 1;
            end
        end
    end

end