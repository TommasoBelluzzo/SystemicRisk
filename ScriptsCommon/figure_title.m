% [INPUT]
% str = A string representing the figure title.
%
% [OUTPUT]
% t   = The handler of the figure title.

function t = figure_title(varargin)

    persistent p;

    if (isempty(p))
        p = inputParser();
        p.addRequired('str',@(x)validateattributes(x,{'char','string'},{'scalartext','nonempty'}));
    end

    p.parse(varargin{:});

    res = p.Results;
    str = res.str;

    if (nargout == 0)
        figure_title_internal(str);
    else
        t = figure_title_internal(str);
    end

end

function t = figure_title_internal(str)

    fig = gcf();
    fig_fts = get(fig,'DefaultAxesFontSize') + 4;
    fig_uni = get(fig,'Units');
    
    if (~strcmp(fig_uni,'pixels'))
        set(fig,'Units','pixels');
        fig_pos = get(fig,'Position');
        set(fig,'Units',fig_uni);
    else
        fig_pos = get(fig,'Position');
    end

    ff = ((fig_fts - 4) * 6.35) / fig_pos(4);

    tit = NaN;
    y_max = 0;
    y_min = 1;

    h = findobj(fig,'Type','axes');
    h_len = length(h);
    h_pos = zeros(h_len,4);

    for i = 1:h_len
        h_cur = h(i);
        
        fig_pos = get(h_cur,'Position');
        h_pos(i,:) = fig_pos;

        if (~strcmp(get(h_cur,'Tag'),'suptitle'))
            fig_y = fig_pos(2);
            fig_hei = fig_pos(4);
            
            if (fig_y < y_min)
                y_min = fig_y - (ff / 15);
            end

            if ((fig_hei + fig_y) > y_max)
                y_max = fig_hei + fig_y + (ff / 10);
            end
        else
            tit = h_cur;
        end
    end

    if (y_max > 0.92)
        scl = (0.92 - y_min) / (y_max - y_min);

        for i = 1:h_len
            fig_pos = h_pos(i,:);
            fig_pos(2) = ((fig_pos(2) - y_min) * scl) + y_min;
            fig_pos(4) = (fig_pos(4) * scl) - ((1 - scl) * (ff / 15));

            set(h(i),'Position',fig_pos);
        end
    end

    np = get(fig,'NextPlot');
    set(fig,'NextPlot','add');

    if (ishghandle(tit))
        delete(tit);
    end

    axes('Position',[0 1 1 1],'Tag','suptitle','Visible','off');
    t_int = text(0.50,-0.05,str,'HorizontalAlignment','center','FontSize',fig_fts);

    set(fig,'NextPlot',np);

    axes(gca());

    if (nargout == 1)
        t = t_int;
    end

end