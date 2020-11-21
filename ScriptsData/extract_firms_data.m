% [INPUT]
% ds = A structure representing the dataset.
% time_series = A string or a cell array of strings defining the time series to extract.
%
% [OUTPUT]
% firms_data = A vector containing n t-by-k float matrices, where n is the number of firms, t is the number of observations and k is the number of time series.

function firms_data = extract_firms_data(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('ds',@(x)validateattributes(x,{'struct'},{'nonempty'}));
        ip.addRequired('time_series',@(x)validateattributes(x,{'cell' 'char'},{'nonempty' 'size' [1 NaN]}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    ds = validate_dataset(ipr.ds);
    time_series = validate_time_series(ds,ipr.time_series);

    nargoutchk(1,1);

    firms_data = extract_firms_data_internal(ds,time_series);

end

function firms_data = extract_firms_data_internal(ds,time_series)

    k = numel(time_series);
    n = ds.N;
    t = ds.T;

    firms_data = cell(n,1);

    for i = 1:n
        firm_data = zeros(t,k);

        for j = 1:numel(time_series)
            ts = ds.(time_series{j});
            firm_data(:,j) = ts(:,i);
        end

        firms_data{i} = firm_data;
    end

end

function time_series = validate_time_series(ds,time_series)

    if (ischar(time_series))
        time_series = {time_series};
    end

    if (any(cellfun(@(x)~ischar(x)||isempty(x),time_series)))
        error('The ''time_series'' parameter contains invalid values.');
    end

    if (any(any(~ismember(time_series,ds.TimeSeries))))
        error(['The ''time_series'' parameter contains non-existent time series. Valid time series are: ''' ds.TimeSeries{1} '''' sprintf(', ''%s''', ds.TimeSeries{2:end}) '.']);
    end

    for i = 1:numel(time_series)
        if (isempty(ds.(time_series{i})))
            error(['The time series ''' time_series{i} ''' is empty and cannot be extracted.']);
        end
    end

end
