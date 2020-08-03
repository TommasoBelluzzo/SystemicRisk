% [INPUT]
% t = A string representing the figure title.
%
% [OUTPUT]
% t = The handler of the figure title.

function t = figure_title(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('s',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1,NaN]}));
    end

    ip.parse(varargin{:});
    ipr = ip.Results;
    
    nargoutchk(0,1);

    if (nargout == 0)
        figure_title_internal(ipr.s);
    else
        t = figure_title_internal(ipr.s);
    end

end

function t = figure_title_internal(s)

    f = gcf();
    f_font_size = get(f,'DefaultAxesFontSize') + 4;
    f_units = get(f,'Units');
    
    if (~strcmp(f_units,'pixels'))
        set(f,'Units','pixels');
        f_position = get(f,'Position');
        set(f,'Units',f_units);
    else
        f_position = get(f,'Position');
    end

    adjustment = ((f_font_size - 4) * 6.35) / f_position(4);
    title = NaN;
    y_max = 0;
    y_min = 1;

    handles = findobj(f,'Type','axes');
    handles_len = length(handles);
    handles_position = zeros(handles_len,4);

    for i = 1:handles_len
        h_current = handles(i);
        
        f_position = get(h_current,'Position');
        handles_position(i,:) = f_position;

        if (~strcmp(get(h_current,'Tag'),'FigureTitle'))
            f_y = f_position(2);
            f_height = f_position(4);
            
            if (f_y < y_min)
                y_min = f_y - (adjustment / 15);
            end

            if ((f_height + f_y) > y_max)
                y_max = f_height + f_y + (adjustment / 10);
            end
        else
            title = h_current;
        end
    end

    if (y_max > 0.92)
        scale = (0.92 - y_min) / (y_max - y_min);

        for i = 1:handles_len
            f_position = handles_position(i,:);
            f_position(2) = ((f_position(2) - y_min) * scale) + y_min;
            f_position(4) = (f_position(4) * scale) - ((1 - scale) * (adjustment / 15));

            set(handles(i),'Position',f_position);
        end
    end

    next_plot = get(f,'NextPlot');
    set(f,'NextPlot','add');

    if (ishghandle(title))
        delete(title);
    end

    axes('Position',[0 1 1 1],'Tag','FigureTitle','Visible','off');
    t_internal = text(0.5000,-0.0157,s,'FontSize',f_font_size,'HorizontalAlignment','center');

    set(f,'NextPlot',next_plot);

    axes(gca());

    if (nargout == 1)
        t = t_internal;
    end

end
