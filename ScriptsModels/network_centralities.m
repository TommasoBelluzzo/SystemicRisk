% [INPUT]
% am = A binary n-by-n matrix {0;1} representing the adjcency matrix.
%
% [OUTPUT]
% bc = A row vector of floats [0,Inf) of length n representing the betweenness centrality values.
% cc = A row vector of floats [0,Inf) of length n representing the closeness centrality values.
% dc = A row vector of floats [0,Inf) of length n representing the degree centrality values.
% ec = A row vector of floats [0,Inf) of length n representing the eigenvector centrality values.
% kc = A row vector of floats [0,Inf) of length n representing the Katz centrality values.
% clc = A row vector of floats [0,Inf) of length n representing the clustering coefficient.
% deg = A row vector of floats [0,Inf) of length n representing the degrees.
% deg_in = A row vector of floats [0,Inf) of length n representing the in-degrees.
% deg_out = A row vector of floats [0,Inf) of length n representing the out-degrees.

function [bc,cc,dc,ec,kc,clc,deg,deg_in,deg_out] = network_centralities(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('am',@(x)validateattributes(x,{'double'},{'real' 'finite' 'binary' '2d' 'square' 'nonempty'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    am = validate_input(ipr.am);

    nargoutchk(9,9);

    [bc,cc,dc,ec,kc,clc,deg,deg_in,deg_out] = network_centralities_internal(am);

end

function [bc,cc,dc,ec,kc,clc,deg,deg_in,deg_out] = network_centralities_internal(am)

    am_len = size(am,1);

    bc = betweenness_centrality(am,am_len);
    [deg,deg_in,deg_out,dc] = degree_centrality(am,am_len);
    cc = closeness_centrality(am,am_len);
    ec = eigenvector_centrality(am);
    kc = katz_centrality(am,am_len);
    clc = clustering_coefficient(am,am_len,deg);

end

function bc = betweenness_centrality(am,am_len)

    bc = zeros(1,am_len);

    for i = 1:am_len
        depth = 0;
        nsp = accumarray([1 i],1,[1 am_len]);
        bfs = false(250,am_len);
        fringe = am(i,:);

        while ((nnz(fringe) > 0) && (depth <= 250))
            depth = depth + 1;
            nsp = nsp + fringe;
            bfs(depth,:) = logical(fringe);
            fringe = (fringe * am) .* ~nsp;
        end

        [rows,cols,v] = find(nsp);
        v = 1 ./ v;

        nsp_inv = accumarray([rows.' cols.'],v,[1 am_len]);

        bcu = ones(1,am_len);

        for depth = depth:-1:2
            w = (bfs(depth,:) .* nsp_inv) .* bcu;
            bcu = bcu + ((am * w.').' .* bfs(depth-1,:)) .* nsp;
        end

        bc = bc + sum(bcu,1);
    end

    bc = bc - am_len;
    bc = (bc .* 2) ./ ((am_len - 1) * (am_len - 2));

end

function cc = closeness_centrality(am,am_len)

    cc = zeros(1,am_len);

    for i = 1:am_len
        paths = dijkstra_shortest_paths(am,am_len,i);
        paths_sum = sum(paths(~isinf(paths)));

        if (paths_sum ~= 0)
            cc(i) = 1 / paths_sum;
        end
    end

    cc = cc .* (am_len - 1);

end

function clc = clustering_coefficient(am,am_len,deg)

    if (issymmetric(am))
        f = 2;
    else
        f = 1;
    end

    clc = zeros(am_len,1);

    for i = 1:am_len
        degree = deg(i);

        if ((degree == 0) || (degree == 1))
            continue;
        end

        k_neighbors = find(am(i,:) ~= 0);
        k_subgraph = am(k_neighbors,k_neighbors);

        if (issymmetric(k_subgraph))
            k_subgraph_trace = trace(k_subgraph);

            if (k_subgraph_trace == 0)
                edges = sum(sum(k_subgraph)) / 2; 
            else
                edges = ((sum(sum(k_subgraph)) - k_subgraph_trace) / 2) + k_subgraph_trace;
            end
        else
            edges = sum(sum(k_subgraph));
        end

        clc(i) = (f * edges) / (degree * (degree - 1));     
    end

    clc = clc.';

end

function [deg,deg_in,deg_out,dc] = degree_centrality(am,am_len)

    deg_in = sum(am);
    deg_out = sum(am.');

    if (issymmetric(am))
        deg = deg_in + diag(am).';
    else
        deg = deg_in + deg_out;
    end

    dc = deg ./ (am_len - 1);

end

function ec = eigenvector_centrality(am)

    [eigen_vector,eigen_values] = eig(am);
    [~,indices] = max(diag(eigen_values));

    ec = abs(eigen_vector(:,indices)).';
    ec = ec ./ sum(ec);

end

function kc = katz_centrality(am,am_len)

    kc = linsolve(eye(am_len) - (am .* 0.1),ones(am_len,1));
    kc = kc.' ./ (sign(sum(kc)) * norm(kc,'fro'));

end

function paths = dijkstra_shortest_paths(am,am_len,node)

    paths = Inf(1,am_len);
    paths(node) = 0;

    s = 1:am_len;

    while (~isempty(s))
        [~,idx] = min(paths(s));
        s_min = s(idx);

        for i = 1:length(s)
            s_i = s(i);

            offset = am(s_min,s_i);
            offset_sum = offset + paths(s_min);

            if ((offset > 0) && (paths(s_i) > offset_sum))
                paths(s_i) = offset_sum;
            end
        end

        s = setdiff(s,s_min);
    end

end

function am = validate_input(am)

    amv = am(:);

    if (numel(amv) < 4)
        error('The value of ''am'' is invalid. Expected input to be a square matrix with a minimum size of 2x2.');
    end

end
