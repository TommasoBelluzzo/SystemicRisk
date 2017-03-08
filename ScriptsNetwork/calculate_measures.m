% [INPUT]
% adjm  = An n-by-n matrix representing the adjacency matrix of the network.
% grp   = A column vector whose values represent the delimiters betweeen the different financial sectors.
%         For example, for the vector [2 7] there will be 3 different sectors:
%          - sector 1 with firms 1,2
%          - sector 2 with firms 3,4,5,6
%          - sector 3 with firms 7,8,…,n
%
% [OUTPUT]
% dci   = A scalar representing the Dynamic Causality Index value.
% n_io  = A scalar representing the total number of in and out connections.
% n_ioo = A scalar representing the total number of in and out connections between different financial sectors.
% cloc  = A column vector containing the normalized closeness centrality of each node.
% degc  = A column vector containing the normalized degree centrality of each node.
% eigc  = A column vector containing the normalized eigenvector centrality of each node.
% clus  = A column vector containing the normalized clustering coefficient of each node.
%
% [NOTE]
% If no sector delimiters are specified, n_ioo is returned as 0.

function [dci,n_io,n_ioo,cloc,degc,eigc,cluc] = calculate_measures(adjm,grp)

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
    
    if (nargin < 2)
        n_ioo = 0;
    else
        grp_len = length(grp);
        n_ifo = zeros(n,1);
        n_oto = zeros(n,1);
        
        for i = 1:n
            grp_1 = grp(1);
            grp_n = grp(grp_len);
            
            if (i <= grp_1)
                grp_beg = 1;
                grp_end = grp_1;
            elseif (i > grp_n)
                grp_beg = grp_n + 1;
                grp_end = n;
            else
                for j = 1:grp_len-1
                    grp_j0 = grp(j);
                    grp_j1 = grp(j+1);

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
    
    [~,cloc] = calculate_closeness_centrality(adjm);
    [degc_std,degc] = calculate_degree_centrality(adjm);
    [~,eigc] = calculate_eigenvector_centrality(adjm);  
    [~,cluc] = calculate_clustering_coefficient(adjm,degc_std);

end