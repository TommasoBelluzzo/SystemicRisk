% [INPUT]
% adjm     = An n-by-n matrix representing the adjacency matrix of the network.
%
% [OUTPUT]
% cluc     = A column vector containing the clustering coefficients.
% cluc_nor = A column vector containing the normalized clustering coefficients.

function [cluc,cluc_nor] = calculate_clustering_coefficient(adjm,degc)

    n = length(adjm);
    n_list = (1:n);

    degc_max = max(degc);
    
    cluc = zeros(n,1);
    cluc_nor = zeros(n,1);

    for i = 1:n
        degc_i = degc(i);
        
        if ((degc_i == 0) || (degc_i == 1))
            continue;
        end

        node = n_list(logical(adjm(:,i)));
        cluc_i = sum(sum(adjm(node,node))) / degc_i / (degc_i - 1);
        
        cluc(i) = cluc_i;
        cluc_nor(i) = cluc_i * (degc_i / degc_max);
    end

end