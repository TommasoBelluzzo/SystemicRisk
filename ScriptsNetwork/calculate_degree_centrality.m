% [INPUT]
% adjm     = An n-by-n matrix representing the adjacency matrix of the network.
%
% [OUTPUT]
% degc     = A column vector containing the degree centralities.
% degc_nor = A column vector containing the normalized degree centralities.

function [degc,degc_nor] = calculate_degree_centrality(adjm)

    n = length(adjm);
    degc = zeros(n,1);

    for i = 1:n
        degc(i) = sum(adjm(:,i)~=0) + sum(adjm(i,:)~=0);
    end
    
    degc_nor = degc ./ (n - 1);

end