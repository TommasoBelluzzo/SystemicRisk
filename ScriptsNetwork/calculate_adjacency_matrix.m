% [INPUT]
% ret = A t-by-n matrix containing firms log returns.
% sst = The statistical significance threshold for the linear Granger-causality Test.
% rob = A boolean indicating whether to use standard or robust p-values.
%
% [OUTPUT]
% adj_mat = An n-by-n matrix representing the adjacency matrix of the network.

function adj_mat = calculate_adjacency_matrix(ret,sst,rob)

    firms = size(ret,2);
    adj_mat = zeros(firms);

    for i = 1:firms
        r_in = ret(:,i);

        parfor j = 1:firms
            if (i == j)
                continue;
            end
            
            r_out = ret(:,j);
            
            if (rob == 0)
                [pval,~] = linear_granger_causality(r_in, r_out);
            else
                [~,pval] = linear_granger_causality(r_in, r_out);
            end

            if (pval < sst)
                adj_mat(i,j) = 1;
            end
        end
    end

end
