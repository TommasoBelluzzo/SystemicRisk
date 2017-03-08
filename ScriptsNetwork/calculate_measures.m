% [INPUT]
% adjm    = An n-by-n matrix representing the adjacency matrix of the network.
% grp     = A vector whose values represent the delimiters betweeen the different financial sectors.
%           For example, for the vector [2 7] there will be 3 different sectors:
%            - sector 1 with firms 1,2
%            - sector 2 with firms 3,4,5,6
%            - sector 3 with firms 7,8,…,n
%
% [OUTPUT]
% dci     = A scalar representing the Dynamic Causality Index value.
% num_io  = A scalar representing the total number of in and out connections.
% num_ioo = A scalar representing the total number of in and out connections between different financial sectors.
% clo_cen = A column vector containing the closeness centrality of each node.
% deg_cen = A column vector containing the degree centrality of each node.
% eig_cen = A column vector containing the eigenvector centrality of each node.
% clust   = A column vector containing the clustering coefficient of each node.
%
% [NOTE]
% If no sector delimiters are specified, num_ioo is returned as NaN.

function [dci,num_io,num_ioo,clo_cen,deg_cen,eig_cen,clust] = calculate_measures(adjm,grp)

    n = length(adjm);

    rel_cur = sum(sum(adjm));
    rel_max = (n ^ 2) - n;
    dci = rel_cur / rel_max;

    num_in = zeros(n,1);
    num_out = zeros(n,1);
    
    for i = 1:n     
        num_in(i) = sum(adjm(:,i));
        num_out(i) = sum(adjm(i,:));
    end

    num_io = sum(num_in) + sum(num_out);
    
    if (nargin < 2)
        num_ioo = NaN;
    else
        grp_len = length(grp);
        num_ifo = zeros(n,1);
        num_oto = zeros(n,1);
        
        for i = 1:n
            if (i <= grp(1))
                grp_beg = 1;
                grp_end = grp(1);
            elseif (i > grp(grp_len))
                grp_beg = grp(grp_len) + 1;
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

            num_ifo(i) = num_in(i) - sum(adjm(grp_beg:grp_end,i));
            num_oto(i) = num_out(i) - sum(adjm(i,grp_beg:grp_end));
        end

        num_ioo = sum(num_ifo) + sum(num_oto);
    end
    
    [~,clo_cen] = calculate_closeness_centrality(adjm);
    [deg_cen_std,deg_cen] = calculate_degree_centrality(adjm);
    eig_cen = calculate_eigenvector_centrality(adjm);  
    clust = calculate_clustering_coefficient(adjm,deg_cen_std);

end