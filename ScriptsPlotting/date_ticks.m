% [INPUT]
% axes = A vector containing the axes on which the date ticks must be applied.
% ... = Additional parameters used by the built-in datetick function.

function date_ticks(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('axes',@(x)validateattributes(x,{'matlab.graphics.axis.Axes'},{'vector' 'nonempty'}));
    end

    ip.parse(varargin{1});
    ipr = ip.Results;

    nargoutchk(0,0);

    date_ticks_internal(ipr.axes,varargin(2:numel(varargin)));

end

function date_ticks_internal(axes,arguments)

    for i = 1:numel(axes)
        arguments_i = [{axes(i)} arguments{:}];
        datetick(arguments_i{:});
    end

end
