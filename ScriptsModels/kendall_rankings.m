% [INPUT]
% ds = A structure representing the dataset.
% measures = A cell array of strings defining the target measures.
%
% [OUTPUT]
% rc = A column vector of floats of length n representing the value of assets.
% rs = A column vector of floats of length n representing the value of assets.

function [rc,rs] = kendall_rankings(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('ds',@(x)validateattributes(x,{'struct'},{'nonempty'}));
        ip.addRequired('measures',@(x)validateattributes(x,{'cell'},{'vector' 'nonempty'}));
    end

    ip.parse(varargin{:});
    
    ipr = ip.Results;
    ds = validate_dataset(ipr.ds);
    measures = validate_input(ds,ipr.measures);

    nargoutchk(2,2);

    [rc,rs] = kendall_rankings_internal(ds,measures);

end

function [rc,rs] = kendall_rankings_internal(ds,measures)

    t = ds.T;

    m = numel(measures);
    measures_pairs = nchoosek(1:m,2);
    
    rc = zeros(m,m);
    rs = zeros(1,m);

    for i = 1:size(measures_pairs,1)
        pair = measures_pairs(i,:);

        index_1 = pair(1);
        field_1 = strrep(ds.LabelsMeasuresSimple{index_1},' ','');
        measure_1 = ds.(field_1);
        
        index_2 = pair(2);
        field_2 = strrep(ds.LabelsMeasuresSimple{index_2},' ','');
        measure_2 = ds.(field_2);
        
        for j = 1:t
            [~,rank_1] = sort(measure_1(j,:),'ascend');
            [~,rank_2] = sort(measure_2(j,:),'ascend');

            rc(index_1,index_2) = rc(index_1,index_2) + concordance_coefficient(rank_1.',rank_2.');
        end
    end
    
    for i = 1:m
        field = strrep(ds.LabelsMeasuresSimple{i},' ','');
        measure = ds.(field);
        
        for j = t:-1:2
            [~,rank_previous] = sort(measure(j-1,:),'ascend');
            [~,rank_current] = sort(measure(j,:),'ascend');

            rs(i) = rs(i) + concordance_coefficient(rank_current.',rank_previous.');
        end
    end
    
    rc = ((rc + rc.') / t) + eye(m);
    rs = rs ./ (t - 1);

end

function cc = concordance_coefficient(rank_1,rank_2)

    m = [rank_1 rank_2];
    [n,k] = size(m);

    rm = zeros(n,k);

    for i = 1:k
        x_i = m(:,i);
        [~,b] = sortrows(x_i,'ascend');
        rm(b,i) = 1:n;
    end

    rm_sum = sum(rm,2);
    s = sum(rm_sum.^2,1) - ((sum(rm_sum) ^ 2) / n);

    cc = (12 * s) / ((k ^ 2) * (( n^ 3) - n));

end

function measures = validate_input(ds,measures)

    if (~iscellstr(measures)) %#ok<ISCLSTR>
        error('The value of ''measures'' is invalid. Expected input to be a cell array of character vectors.');
    end
    
    m = numel(measures);
    
    if (m < 2)
        error('The value of ''measures'' is invalid. Expected input to contain at least 2 elements.');
    end
    
    measures = strrep(measures,' ','');
    
    for i = 1:m
        if (~isfield(ds,measures{i}))
            error('The dataset does not contain all the specified measures.');
        end
    end
    
end
