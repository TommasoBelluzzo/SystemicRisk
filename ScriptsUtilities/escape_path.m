% [INPUT]
% file_path = A string representing the file path.
%
% [OUTPUT]
% file_path_w = A string representing the escaped file path.

function file_path_w = escape_path(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('file_path',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    file_path = ipr.file_path;

    nargoutchk(1,1);

    file_path_w = escape_path_internal(file_path);

end

function file_path_w = escape_path_internal(file_path)

    if (strcmp(filesep(),'\'))
        file_path_w = strrep(file_path,filesep(),[filesep() filesep()]);
    else
        file_path_w = file_path;
    end

end
