% [INPUT]
% handle = A handle to the plotting function to be executed.

function safe_plot(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('handle',@(x)validateattributes(x,{'function_handle'},{'scalar'}));
    end

    ip.parse(varargin{:});
    ipr = ip.Results;

    nargoutchk(0,0);

    safe_plot_internal(ipr.handle);

end

function safe_plot_internal(handle)

    persistent ids;

    name = func2str(handle);
    name = regexprep(name,'^@\([^)]*\)','');
    name = regexprep(name,'\([^)]*\)$','');

    try
        id = [upper(name) '-' upper(char(java.util.UUID.randomUUID()))];
    catch
        id = randi([0 10000000]);

        while (ismember(id,ids))
            id = randi([0 100000]);
        end

        ids = [ids; id];
        id = [upper(name) '-' sprintf('%08s',num2str(id))];
    end

    try
        handle(id);
    catch e
        delete(findobj('Type','Figure','Tag',id));

        try
            r = getReport(e,'Extended','Hyperlinks','off');
            r = strsplit(r,new_line());
            r = cellfun(@(x)['  ' x],r,'UniformOutput',false);
            r = strjoin(r,new_line());
            r = clear_text(r);

            warning('MATLAB:SystemicRisk',['The following exception occurred in the plotting function ''' name ''':' new_line() r]);
        catch
            warning('MATLAB:SystemicRisk',['The following exception occurred in the plotting function ''' name ''':' new_line() e.message]);
        end
    end

end
