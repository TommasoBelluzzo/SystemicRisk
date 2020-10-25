% [INPUT]
% file = A string representing the full path to the Excel spreadsheet.
% type = A string representing the type of Excel spreadsheet:
%   - 'D' for dataset.
%   - 'T' for template.

function file_sheets = validate_xls(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('file',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addRequired('type',@(x)any(validatestring(x,{'D' 'T'})));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    file = ipr.file;
    type = ipr.type;

    nargoutchk(1,1);

    file_sheets = validate_xls_internal(file,type);

end

function file_sheets = validate_xls_internal(file,type)

    if (strcmp(type,'D'))
        label = 'dataset';
    else
        label = 'template';
    end

    if (exist(file,'file') == 0)
        error(['The ' label ' file ''' file ''' could not be found.']);
    end

    [~,~,extension] = fileparts(file);

    if (~strcmp(extension,'.xlsx'))
        error(['The ' label ' file ''' file ''' is not a valid Excel spreadsheet.']);
    end

    if (verLessThan('MATLAB','9.7'))
        if (ispc())
            check_format = true;
            
            try
                [file_status,file_sheets,file_format] = xlsfinfo(file);
            catch
                [file_status,file_sheets] = xlsfinfo(file);
                file_format = [];
                
                check_format = false;
            end

            if (isempty(file_status) || (check_format && ~strcmp(file_format,'xlOpenXMLWorkbook')))
                error(['The ' label ' file ''' file ''' is not a valid Excel spreadsheet.']);
            end
        else
            [file_status,file_sheets] = xlsfinfo(file);

            if (isempty(file_status))
                error(['The ' label ' file ''' file ''' is not a valid Excel spreadsheet.']);
            end
        end
    else
        try
            file_sheets = sheetnames(file);
        catch
            error(['The ' label ' file ''' file ''' is not a valid Excel spreadsheet.']);
        end
    end

end
