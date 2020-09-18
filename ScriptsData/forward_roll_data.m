% [INPUT]
% data = A float t-by-n matrix containing the daily time series to be rolled forward.
% dates = A vector of length t containing the numeric reference dates of observations.
% fr = An integer [0,6] representing the number of months of forward-rolling to be applied to the time series.
%
% [OUTPUT]
% ts = A float t-by-n matrix containing the original time series in which distressed observations are replaced with NaNs.

function data = forward_roll_data(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('data',@(x)validateattributes(x,{'double'},{'real' '2d'}));
        ip.addRequired('dates',@(x)validateattributes(x,{'double'},{'real' 'vector'}));
        ip.addRequired('fr',@(x)validateattributes(x,{'double'},{'real' 'finite' 'integer' '>=' 0 '<=' 6 'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    [data,dates] = validate_input(ipr.data,ipr.dates);
    fr = ipr.fr;

    nargoutchk(1,1);

    data = forward_roll_data_internal(data,dates,fr);

end

function data = forward_roll_data_internal(data,dates,fr)

    if (isempty(data) || (fr == 0))
        return;
    end

    [~,a] = unique(cellstr(datestr(dates,'mm/yyyy')),'stable');
    data_monthly = data(a,:);

    indices_seq = [a(1:fr:numel(a)) - 1; numel(dates)];
    data_seq = data_monthly(1:fr:numel(a),:);

    data_fr = NaN(size(data));

    for i = 2:numel(indices_seq)
        indices = (indices_seq(i-1) + 1):indices_seq(i);
        data_fr(indices,:) = repmat(data_seq(i-1,:),numel(indices),1);
    end
    
    data = data_fr;

end

function [data,dates] = validate_input(data,dates)

    if (~isempty(data))
        t = size(data,1);
        td = numel(dates);

        if (t ~= td)
            error(['The number of rows in the time series (' num2str(t) ') must be equal to the number of dates (' num2str(td) ').']);
        end
    end
    
    dates = dates(:);

end
