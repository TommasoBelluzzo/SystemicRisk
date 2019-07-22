% [INPUT]
% adjm    = A numeric n-by-n matrix representing the adjacency matrix of the network.
% grps    = A vector whose values represent the delimiters betweeen the different firm sectors.
%           For example, the vector [2 7] defines 3 different sectors:
%            - sector 1 with firms 1, 2
%            - sector 2 with firms 3, 4, 5, 6, 7
%            - sector 3 with firms 8, ..., n
%
% [OUTPUT]
% dci   = A float representing the Dynamic Causality Index value.
% n_io  = An integer representing the total number of in and out connections.
% n_ioo = An integer representing the total number of in and out connections between the different sectors (if no groups are specified a NaN is returned).
% betc  = A vector of floats containing the normalized betweenness centrality of each node.
% cloc  = A vector of floats containing the normalized closeness centrality of each node.
% cluc  = A vector of floats containing the normalized clustering coefficient of each node.
% degc  = A vector of floats containing the normalized degree centrality of each node.
% eigc  = A vector of floats containing the normalized eigenvector centrality of each node.
% eigc  = A vector of floats containing the normalized Katz centrality of each node.
%
% [NOTES]
% If no sector delimiters are specified, n_ioo is equal to NaN.

function [dci,n_io,n_ioo,betc,cloc,cluc,degc,eigc,katc] = calculate_measures(adjm,grps)

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
    
    adjm_len = length(adjm);

    betc = calculate_betweenness_centrality(adjm,adjm_len);
    [degc_std,degc] = calculate_degree_centrality(adjm,adjm_len);
    cloc = calculate_closeness_centrality(adjm,adjm_len);
    cluc = calculate_clustering_coefficient(adjm,adjm_len,degc_std);
    eigc = calculate_eigenvector_centrality(adjm);
    katc = calculate_katz_centrality(adjm,adjm_len);

end

function betc = calculate_betweenness_centrality(adjm,adjm_len)

    betc = zeros(1,adjm_len);

    for i = 1:adjm_len
        depth = 0;
        nsp = accumarray([1 i],1,[1 adjm_len]);
        bfs = false(250,adjm_len);
        fringe = adjm(i,:);

        while ((nnz(fringe) > 0) && (depth <= 250))
            depth = depth + 1;
            nsp = nsp + fringe;
            bfs(depth,:) = logical(fringe);
            fringe = (fringe * adjm) .* ~nsp;
        end

        [rows,cols,v] = find(nsp);
        v = 1 ./ v;
        
        nsp_inv = accumarray([rows.' cols.'],v,[1 adjm_len]);

        bcu = ones(1,adjm_len);

        for depth = depth:-1:2
            w = (bfs(depth,:) .* nsp_inv) .* bcu;
            bcu = bcu + ((adjm * w.').' .* bfs(depth-1,:)) .* nsp;
        end

        betc = betc + sum(bcu,1);
    end

    betc = betc - adjm_len;
    betc = (betc .* 2) ./ ((adjm_len - 1) * (adjm_len - 2));

end

function cloc = calculate_closeness_centrality(adjm,adjm_len)

    cloc = zeros(1,adjm_len);

    for i = 1:adjm_len
        dsp = dijkstra_shortest_paths(adjm,adjm_len,i);
        dsp_sum = sum(dsp(~isinf(dsp)));
        
        if (dsp_sum ~= 0)
            cloc(i) = 1 / dsp_sum;
        end
    end

    cloc = cloc .* (adjm_len - 1);

end

function cluc = calculate_clustering_coefficient(adjm,adjm_len,degc)

    adjm_seq = 1:adjm_len;
    degc_max = max(degc);
    
    cluc = zeros(1,adjm_len);

    for i = adjm_seq
        degc_i = degc(i);
        
        if ((degc_i == 0) || (degc_i == 1))
            continue;
        end

        node = adjm_seq(logical(adjm(:,i)));
        cluc_i = (sum(sum(adjm(node,node))) / degc_i) / (degc_i - 1);

        cluc(i) = cluc_i * (degc_i / degc_max);
    end

end

function [degc,degc_nor] = calculate_degree_centrality(adjm,adjm_len)

    degc = zeros(1,adjm_len);

    for i = 1:adjm_len
        degc(i) = sum(adjm(:,i)~=0) + sum(adjm(i,:)~=0);
    end

    degc_nor = degc ./ (adjm_len - 1);

end

function eigc = calculate_eigenvector_centrality(adjm)

	[eig_vec,eig_val] = eig(adjm);
    [~,idx] = max(diag(eig_val));

    eigc = abs(eig_vec(:,idx))';
    eigc = eigc ./ sum(eigc);

end

function katc = calculate_katz_centrality(adjm,adjm_len)

    katc = (eye(adjm_len) - (adjm .* 0.1)) \ ones(adjm_len,1);
    katc = katc.' ./ (sign(sum(katc)) * norm(katc,'fro'));

end

function dist = dijkstra_shortest_paths(adjm,adjm_len,node)

    dist = Inf(1,adjm_len);
    dist(node) = 0;

    adjm_seq = 1:adjm_len;

    while (~isempty(adjm_seq))
        [~,idx] = min(dist(adjm_seq));
        adjm_seq_idx = adjm_seq(idx);

        for i = 1:length(adjm_seq)
            adjm_seq_i = adjm_seq(i);

            adjm_off = adjm(adjm_seq_idx,adjm_seq_i);
            sum_off = adjm_off + dist(adjm_seq_idx);
            
            if ((adjm_off > 0) && (dist(adjm_seq_i) > sum_off))
                dist(adjm_seq_i) = sum_off;
            end
        end

        adjm_seq = setdiff(adjm_seq,adjm_seq_idx);
    end

end