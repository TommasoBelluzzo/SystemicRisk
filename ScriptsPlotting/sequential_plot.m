% [INPUT]
% core = A structure containing the plot data, functions and parameters.
% id = A cell array of strings representing the dynamic plot labels, one for each time series (optional, default='').

function sequential_plot(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('core',@(x)validateattributes(x,{'struct'},{'nonempty'}));
        ip.addOptional('id','',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    core = validate_core(ipr.core);
    
    nargoutchk(0,0);

    sequential_plot_internal(core,ipr.id);

end

function sequential_plot_internal(core,id)

    f = figure('Name',core.OuterTitle,'Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    button_prev = uicontrol('Parent',f,'String','Previous','Style','pushbutton','Callback',@switch_time_series);
    set(button_prev,'Enable','off','FontSize',11,'Units','normalized','Position',[0.01 0.94 0.05 0.04]);
    
    button_next = uicontrol('Parent',f,'String','Next','Style','pushbutton','Callback',@switch_time_series);
    set(button_next,'Enable','on','FontSize',11,'Units','normalized','Position',[0.94 0.94 0.05 0.04]);

    plots_span = core.PlotsSpan;
    plots_rows = core.PlotsAllocation(1);
    plots_columns = core.PlotsAllocation(2);
    plots = numel(plots_span);
    
    subs = gobjects(plots,1);
    
    for i = 1:plots
        subs(i) = subplot(plots_rows,plots_columns,core.PlotsSpan{i});
    end

    core.Function(subs,core.Data(:,1));
    
    for i = 1:plots
        sub = subs(i);
        
        x_label = core.XLabel{i};
        
        if (~isempty(x_label))
             xlabel(sub,x_label);
        end
        
        y_label = core.YLabel{i};
        
        if (~isempty(y_label))
             ylabel(sub,y_label);
        end
        
        x_limits = core.XLimits{i};
        
        if (~isempty(x_limits))
            set(sub,'XLim',x_limits);
        end

        y_limits = core.YLimits{i};
        
        if (~isempty(y_limits))
            set(sub,'YLim',y_limits);
        end
        
        if (core.XGrid{i})
            set(sub,'XGrid','on');
        end
        
        if (core.YGrid{i})
            set(sub,'YGrid','on');
        end

        x_tick = core.XTick{i};

        if (~isempty(x_tick))
            set(sub,'XTick',x_tick);
        end
        
        y_tick = core.YTick{i};

        if (~isempty(y_tick))
            set(sub,'YTick',y_tick);
        end
        
        x_ticklabels = core.XTickLabels{i};

        if (~isempty(x_ticklabels))
            if (isa(x_ticklabels,'function_handle'))
                set(sub,'XTickLabels',arrayfun(@(x)x_ticklabels(x),get(sub,'XTick'),'UniformOutput',false));
            else
                set(sub,'XTickLabels',x_ticklabels);
            end
        end
        
        y_ticklabels = core.YTickLabels{i};

        if (~isempty(y_ticklabels))
            if (isa(y_ticklabels,'function_handle'))
                set(sub,'YTickLabels',arrayfun(@(x)y_ticklabels(x),get(sub,'YTick'),'UniformOutput',false));
            else
                set(sub,'YTickLabels',y_ticklabels);
            end
        end
        
        x_rotation = core.XRotation{i};

        if (~isempty(x_rotation))
            set(sub,'XTickLabelRotation',x_rotation);
        end
        
        y_rotation = core.YRotation{i};

        if (~isempty(y_rotation))
            set(sub,'YTickLabelRotation',y_rotation);
        end
        
        x_dates = core.XDates{i};

        if (~isempty(x_dates))
            if (x_dates)
                date_ticks(sub,'x','mm/yyyy','KeepLimits','KeepTicks');
            else
                date_ticks(sub,'x','yyyy','KeepLimits');
            end
        end

        t = title(sub,core.PlotsTitle{i,1});
        set(t,'Units','normalized');

        if (core.PlotsAdjustable(i))
            t_position = get(t,'Position');
            set(t,'Position',[0.4783 t_position(2) t_position(3)]);
        end
    end

    figure_title([core.InnerTitle ' - ' core.SequenceTitles{1}]);

    setappdata(f,'N',core.N);
    setappdata(f,'Data',core.Data);
    setappdata(f,'Function',core.Function);
    setappdata(f,'InnerTitle',core.InnerTitle);
    setappdata(f,'SequenceTitles',core.SequenceTitles);
    setappdata(f,'PlotsAdjustable',core.PlotsAdjustable);
    setappdata(f,'PlotsTitle',core.PlotsTitle);
    setappdata(f,'Offset',1);
    setappdata(f,'Subs',subs);

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function core = validate_core(core)

    n = validate_field(core,'N',{'double'},{'real' 'finite' 'integer' '>=' 3 'scalar'},{},{});
    validate_field(core,'Data',{'cell'},{'2d' 'nonempty' 'size' [NaN n]},{'double'},{'real' 'nonempty'});
    validate_field(core,'Function',{'function_handle'},{'scalar'},{},{});

    validate_field(core,'OuterTitle',{'char'},{'nonempty' 'size' [1 NaN]},{},{});
    validate_field(core,'InnerTitle',{'char'},{'nonempty' 'size' [1 NaN]},{},{});
    validate_field(core,'SequenceTitles',{'cellstr'},{'nonempty' 'size' [1 n]},{},{});
    
    plots_allocation = validate_field(core,'PlotsAllocation',{'double'},{'real' 'finite' 'integer' '>=' 1 'size' [1 2]},{},{});
    plots_span = validate_field(core,'PlotsSpan',{'cell'},{'row' 'nonempty'},{'double'},{'real' 'finite' 'integer' 'increasing' '>=' 1 'size' [1 NaN]});
    plots = numel(plots_span);
    core.PlotsAdjustable = validate_plots_structure(plots_allocation,plots_span);
    
    validate_field(core,'PlotsTitle',{'cellstr'},{'nonempty' 'size' [plots n]},{},{});

    x_dates = validate_field(core,'XDates',{'cell'},{'nonempty' 'size' [1 plots]},{'logical'},{'optional' 'scalar'});
    validate_field(core,'XGrid',{'cell'},{'nonempty' 'size' [1 plots]},{'logical'},{'scalar'});
    validate_field(core,'XLabel',{'cell'},{'nonempty' 'size' [1 plots]},{'char'},{'optional' 'nonempty' 'size' [1 NaN]});
    validate_field(core,'XLimits',{'cell'},{'nonempty' 'size' [1 plots]},{'double'},{'optional' 'real' 'finite' 'nonempty' 'size' [NaN 2]});
    validate_field(core,'XRotation',{'cell'},{'nonempty' 'size' [1 plots]},{'double'},{'optional' 'real' 'finite' '>=' 0 '<=' 360 'scalar'});
    validate_field(core,'XTick',{'cell'},{'nonempty' 'size' [1 plots]},{'double'},{'optional' 'real' 'finite' 'nonempty' 'size' [1 NaN]});
    x_ticklabels = validate_field(core,'XTickLabels',{'cell'},{'nonempty' 'size' [1 plots]},{'cellfun'},{'optional' 'nonempty' 'size' [1 NaN]});
    
    for i = 1:plots
        x_dates_i = x_dates{i};
        x_ticklabels_i = x_ticklabels{i};
        
        if (~isempty(x_dates_i) && ~isempty(x_ticklabels_i))
            error('The structure cannot define both ''XDates'' and ''XTickLabels'' fields content for the same plot.');
        end
    end
    
    validate_field(core,'YGrid',{'cell'},{'nonempty' 'size' [1 plots]},{'logical'},{'scalar'});
    validate_field(core,'YLabel',{'cell'},{'nonempty' 'size' [1 plots]},{'char'},{'optional' 'nonempty' 'size' [1 NaN]});
    validate_field(core,'YLimits',{'cell'},{'nonempty' 'size' [1 plots]},{'double'},{'optional' 'real' 'finite' 'nonempty' 'size' [NaN 2]});
    validate_field(core,'YRotation',{'cell'},{'nonempty' 'size' [1 plots]},{'double'},{'optional' 'real' 'finite' '>=' 0 '<=' 360 'scalar'});
    validate_field(core,'YTick',{'cell'},{'nonempty' 'size' [1 plots]},{'double'},{'optional' 'real' 'finite' 'nonempty' 'size' [1 NaN]});
    validate_field(core,'YTickLabels',{'cell'},{'nonempty' 'size' [1 plots]},{'cellfun'},{'optional' 'nonempty' 'size' [1 NaN]});

end

function plots_adjustable = validate_plots_structure(plots_allocation,plots_span)

    plots = numel(plots_span);
    plots_rows = plots_allocation(1);
    plots_columns = plots_allocation(2);
    plots_grid = repmat(1:plots_columns,plots_rows,1) + repmat(plots_columns .* (0:plots_rows-1),plots_columns,1).';

    plots_adjustable = false(plots,1);
    plots_slots = cell(plots,1);
    
    for i = 1:plots
        plot_span = plots_span{i};
        
        if (isscalar(plot_span))
            [check,indices] = ismember(plot_span,plots_grid);
            
            if (~check)
                error(['The structure field ''PlotsSpan'' is invalid.' newline() 'Expected content ' num2str(i) ' to define valid grid cells.']);
            end
            
            plot_cells = false(size(plots_grid));
            plot_cells(indices) = true;
            
            rc = 1;
        else
            if ((numel(plot_span) == 2) && (plot_span(1) ~= (plot_span(2) - 1)))
                [r1,c1] = find(plots_grid == plot_span(1),1,'first');

                if (isempty(r1) || isempty(c1))
                    error(['The structure field ''PlotsSpan'' is invalid.' newline() 'Expected content ' num2str(i) ' to define valid grid cells.']);
                end

                [r2,c2] = find(plots_grid == plot_span(2),1,'first');

                if (isempty(r2) || isempty(c2))
                    error(['The structure field ''PlotsSpan'' is invalid.' newline() 'Expected content ' num2str(i) ' to define valid grid cells.']);
                end

                plot_cells = (plots_grid == plot_span(1)) + (plots_grid == plot_span(2));
                plot_cells(r1:r2,c1:c2) = 1;
                plot_cells = logical(plot_cells);

                rc = sum(plot_cells,2);
            else
                [check,indices] = ismember(plot_span,plots_grid);

                if (~all(check))
                    error(['The structure field ''PlotsSpan'' is invalid.' newline() 'Expected content ' num2str(i) ' to define valid grid cells.']);
                end

                if (~all(diff(plot_span) == 1))
                    error(['The structure field ''PlotsSpan'' is invalid.' newline() 'Expected content ' num2str(i) ' to define contiguous grid cells.']);
                end

                plot_cells = false(size(plots_grid));
                plot_cells(indices) = true;

                rc = sum(plot_cells,2);
                rc(rc == 0) = [];

                if (~isscalar(rc))
                    [~,indices_a,indices_b] = intersect(plot_cells,plot_cells,'rows');

                    if (~isscalar(indices_a) || ~isscalar(indices_b) || (indices_a ~= indices_b) || ~all(diff(rc) == 0))
                        error(['The structure field ''PlotsSpan'' is invalid.' newline() 'Expected content ' num2str(i) ' to define a valid subplot area.']);
                    end
                end
            end
        end
        
        if (any(rc == plots_columns))
            plots_adjustable(i) = true;
        end

        plot_slots = plots_grid(plot_cells);
        plots_slots{i} = plot_slots(:);
    end
    
    plots_slots = cell2mat(plots_slots);
    
    if (numel(unique(plots_slots)) ~= numel(plots_slots))
        error(['The structure field ''PlotsSpan'' is invalid.' newline() 'Expected contents to define a non-overlapping subplot areas.']);
    end

end

function validate_content(field_name,content,content_type,content_validator)

    content_iscellfun = (numel(content_type) == 1) && strcmp(content_type{1},'cellfun');
    content_iscellstr = (numel(content_type) == 1) && strcmp(content_type{1},'cellstr');
    content_isoptional = strcmp(content_validator{1},'optional');
    
    if (content_isoptional)
        empty = false;

        try
            validateattributes(content,{'double'},{'size',[0 0]});
            empty = true;
        catch
        end

        if (empty)
            return;
        end

        content_validator = content_validator(2:end);
    end
    
    if (content_iscellfun)
        if (isa(content,'function_handle'))
            content_type = {'function_handle'};
            content_validator = {'scalar'};
        elseif (iscellstr(content)) %#ok<ISCLSTR>
            if (any(cellfun(@length,content) == 0))
                error(['The structure field ''' field_name ''' is invalid.' newline() 'Expected contents to be function handles or cell arrays of non-empty character vectors.']);
            end

            content_type{1} = 'cell';
        else
            error(['The structure field ''' field_name ''' is invalid.' newline() 'Expected contents to be function handles or cell arrays of non-empty character vectors.']);
        end
    elseif (content_iscellstr)
        if (~iscellstr(content) || any(any(cellfun(@length,content) == 0))) %#ok<ISCLSTR>
            error(['The structure field ''' field_name ''' is invalid.' newline() 'Expected contents to be cell arrays of non-empty character vectors.']);
        end

        content_type{1} = 'cell';
    end
    
    try
        validateattributes(content,content_type,content_validator);
    catch e
        error(['The structure field ''' field_name ''' is invalid.' newline() strrep(e.message,'Expected input','Expected contents')]);
    end

end

function value = validate_field(data,field_name,field_type,field_validator,content_type,content_validator)

    if (~isfield(data,field_name))
        error(['The structure does not contain the field ''' field_name '''.']);
    end

    value = data.(field_name);
    value_iscell = (numel(field_type) == 1) && strcmp(field_type{1},'cell');
    value_iscellstr = (numel(field_type) == 1) && strcmp(field_type{1},'cellstr');

    if (value_iscell)
        try
            validateattributes(value,field_type,field_validator);
        catch e
            error(['The structure field ''' field_name ''' is invalid.' newline() strrep(e.message,'Expected input','Expected value')]);
        end
        
        if (~isempty(content_type) && ~isempty(content_validator))
            cellfun(@(x)validate_content(field_name,x,content_type,content_validator),value);
        elseif (isempty(content_type) && ~isempty(content_validator)) || (~isempty(content_type) && isempty(content_validator))
            error('The ''content_type'' and ''content_validator'' parameters must be both empty or non-empty.');
        end
    else
        if (~isempty(content_type) || ~isempty(content_validator))
            error('The ''content_type'' and ''content_validator'' parameters can be used only for cell arrays.');
        end
        
        if (value_iscellstr)
            if (~iscellstr(value) || any(any(cellfun(@length,value) == 0))) %#ok<ISCLSTR>
                error(['The structure field ''' field_name ''' is invalid.' newline() 'Expected value to be a cell array of non-empty character vectors.']);
            end

            field_type{1} = 'cell';
        end
        
        try
            validateattributes(value,field_type,field_validator);
        catch e
            error(['The structure field ''' field_name ''' is invalid.' newline() strrep(e.message,'Expected input','Expected value')]);
        end
    end

end

function draw_time_series(f,offset)

    data = getappdata(f,'Data');
    fun = getappdata(f,'Function');
    inner_title = getappdata(f,'InnerTitle');
    sequence_titles = getappdata(f,'SequenceTitles');
    plots_adjustable = getappdata(f,'PlotsAdjustable');
    plots_title = getappdata(f,'PlotsTitle');
    subs = getappdata(f,'Subs');

    subs_len = numel(subs);

    for i = 1:subs_len
        sub = subs(i);
        
        cla(sub);
        hold(sub,'on');
    end

    fun(subs,data(:,offset));

    for i = 1:subs_len
        sub = subs(i);
        
        hold(sub,'off');
        t = title(sub,plots_title{i,offset});
        set(t,'Units','normalized');
        
        if (plots_adjustable(i))
            t_position = get(t,'Position');
            set(t,'Position',[0.4783 t_position(2) t_position(3)]);
        end
    end

    figure_title([inner_title ' - ' sequence_titles{offset}]);

end

function switch_time_series(obj,~)

    f = get(obj,'Parent');

    direction = get(obj,'String');
    offset = getappdata(f,'Offset');
    
    if (strcmp(direction,'Next'))
        offset = offset + 1;
        setappdata(f,'Offset',offset);
        
        obj_other = findobj(f,'String','Previous');
        set(obj_other,'Enable','on');
        
        n = getappdata(f,'N');

        if (offset == n)
            set(obj,'Enable','off');
        end
    else
        offset = offset - 1;
        setappdata(f,'Offset',offset);
        
        obj_other = findobj(f,'String','Next');
        set(obj_other,'Enable','on');

        if (offset == 1)
            set(obj,'Enable','off');
        end
    end
    
    draw_time_series(f,offset);

end
