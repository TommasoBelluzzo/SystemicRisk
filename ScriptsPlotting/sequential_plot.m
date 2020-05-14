% [INPUT]
% core = A structure containing the plot data, functions and parameters.
% id = A cell array of strings representing the dynamic plot labels, one for each time series (optional, default='').

function sequential_plot(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('core',@(x)validateattributes(x,{'struct'},{'nonempty'}));
        ip.addOptional('id','',@(x)validateattributes(x,{'char'},{'size',[NaN NaN]}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    core = validate_core(ipr.core);
    
    nargoutchk(0,0);

	sequential_plot_internal(core,ipr.id);

end

function sequential_plot_internal(core,id)

    f = figure('Name',[core.OuterTitle ' > ' core.InnerTitle],'Units','normalized','Position',[100 100 0.85 0.85],'Tag',id);

    button_prev = uicontrol('Parent',f,'String','Previous','Style','pushbutton','Callback',@switch_time_series);
    set(button_prev,'Enable','off','FontSize',11,'Units','normalized','Position',[0.01 0.94 0.05 0.04]);
    
    button_next = uicontrol('Parent',f,'String','Next','Style','pushbutton','Callback',@switch_time_series);
    set(button_next,'Enable','on','FontSize',11,'Units','normalized','Position',[0.94 0.94 0.05 0.04]);

    subs = gobjects(core.Plots,1);
    
    if (strcmp(core.PlotsType,'H'))
        for i = 1:core.Plots
            subs(i) = subplot(1,core.Plots,i);
        end
    else
        for i = 1:core.Plots
            subs(i) = subplot(core.Plots,1,i);
        end
    end

    core.PlotFunction(subs,core.X,core.SequenceFunction(core.Y,1));
    
    x_limits_len = size(core.XLimits,1);
    x_limits_mul = (x_limits_len > 1) && (x_limits_len == core.Plots);
    
    y_limits_len = size(core.YLimits,1);
	y_limits_mul = (y_limits_len > 1) && (y_limits_len == core.Plots);
    
    for i = 1:numel(subs)
        sub = subs(i);
        
        if (~isempty(core.XLabel))
             xlabel(sub,core.XLabel);
        end
        
        if (~isempty(core.YLabel))
             ylabel(sub,core.YLabel);
        end

        if (x_limits_mul)
            set(sub,'XLim',core.XLimits(i,:));
        else
            set(sub,'XLim',core.XLimits);
        end

        if (y_limits_mul)
            set(sub,'YLim',core.YLimits(i,:));
        else
            set(sub,'YLim',core.YLimits);
        end
        
        if (core.XGrid)
            set(sub,'XGrid','on');
        end
        
        if (core.YGrid)
            set(sub,'YGrid','on');
        end

        if (~isempty(core.XTick))
            set(sub,'XTick',core.XTick);
        end

        if (~isempty(core.YTick))
            set(sub,'YTick',core.YTick);
        end

        if (~isempty(core.XTickLabels))
            if (isa(core.XTickLabels,'function_handle'))
                set(sub,'XTickLabels',arrayfun(@(x)core.XTickLabels(x),get(sub,'XTick'),'UniformOutput',false));
            else
                set(sub,'XTickLabels',core.XTickLabels);
            end
        end

        if (~isempty(core.YTickLabels))
            if (isa(core.YTickLabels,'function_handle'))
                set(sub,'YTickLabels',arrayfun(@(x)core.YTickLabels(x),get(sub,'YTick'),'UniformOutput',false));
            else
                set(sub,'YTickLabels',core.YTickLabels);
            end
        end

        if (~isempty(core.XRotation))
            set(sub,'XTickLabelRotation',core.XRotation);
        end

        if (~isempty(core.YRotation))
            set(sub,'YTickLabelRotation',core.YRotation);
        end

        if (~isempty(core.XDates))
            if (core.XDates)
                date_ticks(sub,'x','mm/yyyy','KeepLimits','KeepTicks');
            else
                date_ticks(sub,'x','yyyy','KeepLimits');
            end
        end

        if (~isempty(core.PlotsTitle))
            t = title(sub,[core.PlotsTitle{i} ' - ' core.PlotsLabels{i,1}]);
        else
            t = title(sub,core.PlotsLabels{i,1});
        end

        set(t,'Units','normalized');
        
        if ((core.Plots == 1) || strcmp(core.PlotsType,'V') || (strcmp(core.PlotsType,'H') && (mod(core.Plots,2) > 0) && (i == (floor(plots / 2) + 1))))
            t_position = get(t,'Position');
            set(t,'Position',[0.4783 t_position(2) t_position(3)]);
        end
    end

    figure_title(core.InnerTitle);

    setappdata(f,'N',core.N);
    setappdata(f,'Offset',1);
    setappdata(f,'PlotFunction',core.PlotFunction);
	setappdata(f,'PlotsLabels',core.PlotsLabels);
    setappdata(f,'PlotsTitle',core.PlotsTitle);
    setappdata(f,'PlotsType',core.PlotsType);
    setappdata(f,'SequenceFunction',core.SequenceFunction);
    setappdata(f,'Subs',subs);
    setappdata(f,'X',core.X);
    setappdata(f,'Y',core.Y);

    pause(0.01);
    frame = get(f,'JavaFrame');
    set(frame,'Maximized',true);

end

function core = validate_core(core)

    n = validate_field(core,'N',{'double'},{'real','finite','integer','>=',3,'scalar'});
    validate_field(core,'PlotFunction',{'function_handle'},{'scalar'});
    validate_field(core,'SequenceFunction',{'function_handle'},{'scalar'});
    
    validate_field(core,'OuterTitle',{'char'},{'nonempty','size',[1 NaN]});
    validate_field(core,'InnerTitle',{'char'},{'nonempty','size',[1 NaN]});

    plots = validate_field(core,'Plots',{'double'},{'real','finite','integer','>=',1,'scalar'});
    plots_labels = validate_field(core,'PlotsLabels',{'cellstr'},{'nonempty','size',[NaN n]});
    validate_field(core,'PlotsTitle',{'cellstr'},{'optional','nonempty','size',[1 plots]});
    validate_field(core,'PlotsType',{'str'},{'optional','H','V'});
    
    plots_labels_rows = size(plots_labels,1);
    
    if (plots_labels_rows == 1)
        if (plots > 1)
            plots_labels = repmat(plots_labels,plots,1);
        end
    else
        if (plots_labels_rows ~= plots)
            error('The structure must define a ''PlotsLabels'' field with the number of rows equal to 1 or ''Plots''.');
        end
    end

    x = validate_field(core,'X',{'double'},{'real','finite','nonempty'});
    x_dates = validate_field(core,'XDates',{'logical'},{'optional','scalar'});
    validate_field(core,'XGrid',{'logical'},{'scalar'});
    validate_field(core,'XLabel',{'char'},{'optional','nonempty','size',[1 NaN]});
    validate_field(core,'XLimits',{'double'},{'real','finite','nonempty','size',[NaN 2]});
    validate_field(core,'XRotation',{'double'},{'optional','real','finite','>=',0,'<=',360,'scalar'});
    validate_field(core,'XTick',{'double'},{'optional','real','finite','nonempty','size',[1 NaN]});
    x_tick_labels = validate_field(core,'XTickLabels',{'cellfun'},{'optional','nonempty','size',[1 n]});
    
    if (~isempty(x_dates) && ~isempty(x_tick_labels))
        error('The structure cannot define both ''XDates'' and ''XTickLabels'' fields.');
    end

    y = validate_field(core,'Y',{'double'},{'real','nonempty'});
    validate_field(core,'YGrid',{'logical'},{'scalar'});
    validate_field(core,'YLabel',{'char'},{'optional','nonempty','size',[1 NaN]});
    validate_field(core,'YLimits',{'double'},{'real','finite','nonempty','size',[NaN 2]});
    validate_field(core,'YRotation',{'double'},{'optional','real','finite','>=',0,'<=',360,'scalar'});
    validate_field(core,'YTick',{'double'},{'optional','real','finite','nonempty','size',[1 NaN]});
    validate_field(core,'YTickLabels',{'cellfun'},{'optional','nonempty','size',[1 n]});
    
    x_size = size(x);
    y_size = size(y);
    
    if (~all(x_size >= 1))
        error(['The structure field ''X'' is invalid.' newline() 'Expected value to have all sizes >= 1.']);
    end

    if (~all(y_size >= 1))
        error(['The structure field ''Y'' is invalid.' newline() 'Expected value to have all sizes >= 1.']);
    end
    
    if (~any(y_size == n))
        error(['The structure field ''Y'' is invalid.' newline() 'Expected value to have one size matching the field ''N''.']);
    end
    
    if (~any(ismember(y_size,x_size)))
        error(['The structure field ''Y'' is invalid.' newline() 'Expected value to have one size matching the number of elements of the field ''X''.']);
    end
    
    core.PlotsLabels = plots_labels;
    
end

function value = validate_field(data,field_name,field_type,field_validator)

    if (~isfield(data,field_name))
        error(['The structure does not contain the field ''' field_name '''.']);
    end

    value = data.(field_name);
    value_iscellfun = (numel(field_type) == 1) && strcmp(field_type{1},'cellfun');
    value_iscellstr = (numel(field_type) == 1) && strcmp(field_type{1},'cellstr');
    value_isoptional = strcmp(field_validator{1},'optional');
    value_isstr = (numel(field_type) == 1) && strcmp(field_type{1},'str');

    if (value_isoptional)
        empty = false;

        try
            validateattributes(value,{'double'},{'size',[0 0]});
            empty = true;
        catch
        end
        
        if (empty)
            return;
        end
        
        field_validator = field_validator(2:end);
    end
    
    if (value_isstr)
        try
            validatestring(value,field_validator);
        catch e
            error(['The structure field ''' field_name ''' is invalid.' newline() strrep(e.message,'Expected input','Expected value')]);
        end
    else
        if (value_iscellfun)
            if (isa(value,'function_handle'))
                field_type = {'function_handle'};
                field_validator = {'scalar'};
            elseif (iscellstr(value)) %#ok<ISCLSTR>
                if (any(cellfun(@length,value) == 0))
                    error(['The structure field ''' field_name ''' is invalid.' newline() 'Expected value to be a function handle or a cell array of non-empty character vectors.']);
                end

                field_type{1} = 'cell';
            else
                error(['The structure field ''' field_name ''' is invalid.' newline() 'Expected value to be a function handle or a cell array of non-empty character vectors.']);
            end
        elseif (value_iscellstr || value_iscellfun)
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

    plots_labels = getappdata(f,'PlotsLabels');
    plot_function = getappdata(f,'PlotFunction');
    plots_title = getappdata(f,'PlotsTitle');
    plots_type = getappdata(f,'PlotsType');
    sequence_function = getappdata(f,'SequenceFunction');
    subs = getappdata(f,'Subs');
    x = getappdata(f,'X');
    y = getappdata(f,'Y');
    
    subs_len = numel(subs);

    for i = 1:subs_len
        cla(subs(i));
        hold(subs(i),'on');
    end

	plot_function(subs,x,sequence_function(y,offset));
    
    for i = 1:subs_len
        hold(subs(i),'off');
        
        if (~isempty(plots_title))
            t = title(subs(i),[plots_title{i} ' - ' plots_labels{i,offset}]);
        else
            t = title(subs(i),plots_labels{i,offset});
        end

        set(t,'Units','normalized');
    
        if ((subs_len == 1) || strcmp(plots_type,'V'))
            t_position = get(t,'Position');
            set(t,'Position',[0.4783 t_position(2) t_position(3)]);
        end
    end

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
