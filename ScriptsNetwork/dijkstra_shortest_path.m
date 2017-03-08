% [INPUT]
% adjm = An n-by-n matrix representing the adjacency matrix of the network.
% node = A scalar representing the start node.
%
% [OUTPUT]
% dist = A scalar representing the shortest path length from the start node to all the other nodes.

function dist = dijkstra_shortest_path(adjm,node)

    n = length(adjm);
    n_seq = (1:n);

    dist = inf * ones(1,n);
    dist(node) = 0;

    while (~isempty(n_seq))
        [~,idx] = min(dist(n_seq));

        for i = 1:length(n_seq)
            n_seq_i = n_seq(i);
            n_seq_idx = n_seq(idx);
            
            adjm_cur = adjm(n_seq_idx,n_seq_i);
            dist_sum = dist(n_seq_idx) + adjm_cur;
            
            if ((adjm_cur > 0) && (dist(n_seq_i) > dist_sum))
                dist(n_seq_i) = dist_sum;
            end
        end
    
        n_seq = setdiff(n_seq,n_seq_idx);
    end

end