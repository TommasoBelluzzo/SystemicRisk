% [INPUT]
% adjm     = An n-by-n matrix representing the adjacency matrix of the network.
%
% [OUTPUT]
% eigc     = A column vector containing the eigenvector centralities.
% eigc_nor = A column vector containing the normalized eigenvector centralities.

function [eigc,eigc_nor] = calculate_eigenvector_centrality(adjm)

    n = length(adjm);

    if n < 1000
        [eig_vec,eig_val] = eig(adjm);
    else
        [eig_vec,eig_val] = eigs(sparse(adjm));
    end

    [~,idx] = max(diag(eig_val));

    eigc = abs(eig_vec(:,idx));
    eigc_nor = eigc / sum(eigc);

end