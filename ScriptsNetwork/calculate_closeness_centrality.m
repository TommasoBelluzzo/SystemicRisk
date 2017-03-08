% [INPUT]
% adjm     = An n-by-n matrix representing the adjacency matrix of the network.
%
% [OUTPUT]
% cloc     = A column vector containing the closeness centralities.
% cloc_nor = A column vector containing the normalized closeness centralities.
%
% [NOTES]
% The closeness centralities are calculated using the Dangalchev Variant (2006) to take into account disconnected graphs.

function [cloc,cloc_nor] = calculate_closeness_centrality(adjm)

    n = length(adjm);
    cloc = zeros(n,1);

    for i = 1:n
        cloc(i) = sum(2 .^ -dijkstra_shortest_path(adjm,i));
    end

    cloc_nor = cloc ./ (n - 1);

end