% [INPUT]
% f = A valid figure instance.

function maximize_figure(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('f',@(x)validateattributes(x,{'matlab.ui.Figure'},{'scalar'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    f = ipr.f;

    nargoutchk(0,0);

    maximize_figure_internal(f);

end

function maximize_figure_internal(f)

    pause(0.1);

    if (verLessThan('MATLAB','9.4'))
        frame = get(f,'JavaFrame'); %#ok<JAVFM> 
        set(frame,'Maximized',true);
    else
        set(f,'WindowState','maximized');
    end

    drawnow();

end
