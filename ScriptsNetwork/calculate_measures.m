% [INPUT]
% adjm  = A numeric n-by-n matrix representing the adjacency matrix of the network.
% grps  = A vector whose values represent the delimiters betweeen the different firm sectors.
%         For example, for the vector [2 7] there will be 3 different sectors:
%          - sector 1 with firms 1,2
%          - sector 2 with firms 3,4,5,6
%          - sector 3 with firms 7,8,...,n
%
% [OUTPUT]
% dci   = A float representing the Dynamic Causality Index value.
% n_io  = An integer representing the total number of in and out connections.
% n_ioo = An integer representing the total number of in and out connections between the different sectors (if no groups are specified a NaN is returned).
% cloc  = A vector of floats containing the normalized closeness centrality of each node.
% cluc  = A vector of floats containing the normalized clustering coefficient of each node.
% degc  = A vector of floats containing the normalized degree centrality of each node.
% eigc  = A vector of floats containing the normalized eigenvector centrality of each node.
%
% [NOTE]
% If no sector delimiters are specified, n_ioo is returned as 0.

function [dci,n_io,n_ioo,degc,cloc,cluc,eigc] = calculate_measures(adjm,grps)

    n = length(adjm);

    rel_cur = sum(sum(adjm));
    rel_max = (n ^ 2) - n;
    dci = rel_cur / rel_max;

    n_i = zeros(n,1);
    n_o = zeros(n,1);
    
    for i = 1:n     
        n_i(i) = sum(adjm(:,i));
        n_o(i) = sum(adjm(i,:));
    end

    n_io = sum(n_i) + sum(n_o);
    
    if (isempty(grps))
        n_ioo = NaN;
    else
        grp_len = length(grps);
        n_ifo = zeros(n,1);
        n_oto = zeros(n,1);
        
        for i = 1:n
            grp_1 = grps(1);
            grp_n = grps(grp_len);
            
            if (i <= grp_1)
                grp_beg = 1;
                grp_end = grp_1;
            elseif (i > grp_n)
                grp_beg = grp_n + 1;
                grp_end = n;
            else
                for j = 1:grp_len-1
                    grp_j0 = grps(j);
                    grp_j1 = grps(j+1);

                    if ((i > grp_j0) && (i <= grp_j1))
                        grp_beg = grp_j0 + 1;
                        grp_end = grp_j1;
                    end
                end
            end

            n_ifo(i) = n_i(i) - sum(adjm(grp_beg:grp_end,i));
            n_oto(i) = n_o(i) - sum(adjm(i,grp_beg:grp_end));
        end

        n_ioo = sum(n_ifo) + sum(n_oto);
    end
    
    if (nargout >= 4)
        adjm_len = length(adjm);
        adjm_seq = 1:adjm_len;
    
        [degc_std,degc] = calculate_degree_centrality(adjm,adjm_len,adjm_seq);
    end
    
    if (nargout >= 5)
        [~,cloc] = calculate_closeness_centrality(adjm,adjm_len,adjm_seq);
    end
    
    if (nargout >= 6)
        [~,cluc] = calculate_clustering_coefficient(adjm,adjm_len,adjm_seq,degc_std);
    end
    
    if (nargout >= 7)
        [~,eigc] = calculate_eigenvector_centrality(adjm,adjm_len);
    end

end

function [cloc,cloc_nor] = calculate_closeness_centrality(adjm,adjm_len,adjm_seq)

    cloc = zeros(1,adjm_len);

    for i = adjm_seq
        cloc(i) = sum(2 .^ -dijkstra_shortest_path(adjm,i));
    end

    cloc_nor = cloc ./ (adjm_len - 1);

end

function [cluc,cluc_nor] = calculate_clustering_coefficient(adjm,adjm_len,adjm_seq,degc)

    degc_max = max(degc);
    
    cluc = zeros(1,adjm_len);
    cluc_nor = zeros(1,adjm_len);

    for i = adjm_seq
        degc_i = degc(i);
        
        if ((degc_i == 0) || (degc_i == 1))
            continue;
        end

        node = adjm_seq(logical(adjm(:,i)));
        cluc_i = (sum(sum(adjm(node,node))) / degc_i) / (degc_i - 1);
        
        cluc(i) = cluc_i;
        cluc_nor(i) = cluc_i * (degc_i / degc_max);
    end

end

function [degc,degc_nor] = calculate_degree_centrality(adjm,adjm_len,adjm_seq)

    degc = zeros(1,adjm_len);

    for i = adjm_seq
        degc(i) = sum(adjm(:,i)~=0) + sum(adjm(i,:)~=0);
    end
    
    degc_nor = degc ./ (adjm_len - 1);

end

function [eigc,eigc_nor] = calculate_eigenvector_centrality(adjm,adjm_len)

    if (adjm_len <= 1000)
        [eig_vec,eig_val] = eig(adjm);
    else
        [eig_vec,eig_val] = eigs(sparse(adjm));
    end

    [~,idx] = max(diag(eig_val));

    eigc = abs(eig_vec(:,idx))';
    eigc_nor = eigc ./ sum(eigc);

end

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