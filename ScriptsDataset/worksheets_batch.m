% [INPUT]
% file = A string representing the full path to the Excel spreadsheet.
% sheets = A cell array of strings defining the target sheets.
% names = A cell array of strings defining the sheets names.
%
% [NOTES]
% The function only works for Windows machines with Excel ActiveX support.
% If the 'names' parameter is provided, spreadsheet items not matching the target sheets are deleted and the other items are renamed.
% If the 'names' parameter is not provided, a full cleanup of the spreadsheet items is performed.

function worksheets_batch(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('file',@(x)validateattributes(x,{'char'},{'nonempty' 'size' [1 NaN]}));
        ip.addRequired('sheets',@(x)validateattributes(x,{'cell'},{'vector' 'nonempty'}));
        ip.addOptional('names',[],@(x)validateattributes(x,{'cell'},{'vector' 'nonempty'}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    file = ipr.file;
    [sheets,names] = validate_input(ipr.sheets,ipr.names);

    nargoutchk(0,0);

    worksheets_batch_internal(file,sheets,names);

end

function worksheets_batch_internal(file,sheets,names)

    if (exist(file,'file') == 0)
        error(['The file ''' file ''' could not be found.']);
    end

    [~,~,extension] = fileparts(file);

    if (~strcmpi(extension,'.xlsx'))
        error(['The file ''' file ''' is not a valid Excel spreadsheet.']);
    end
    
    if (~ispc())
        warning('MATLAB:SystemicRisk','The current machine does not provide Excel ActiveX support.');
        return;
    end

    try
        excel = actxserver('Excel.Application');
    catch
        warning('MATLAB:SystemicRisk','The current machine does not provide Excel ActiveX support.');
        return;
    end

    try
        wb = excel.Workbooks.Open(file,0,false);
        
        if (isempty(names))
            for i = 1:numel(sheets)
                wb.Sheets.Item(sheets{i}).Cells.Clear();
            end
        else
            sheets_unmatched = cell(1000,1);
            off = 1;

            for i = 1:wb.Sheets.Count
                sheet = wb.Sheets.Item(i).Name;

                if (~ismember(sheet,sheets))
                    sheets_unmatched{off} = sheet;
                    off = off + 1;
                end
            end

            sheets_unmatched(off:end,:) = [];
            sheets_unmatched_len = numel(sheets_unmatched);

            if (sheets_unmatched_len > 0)
                for i = 1:sheets_unmatched_len
                    wb.Sheets.Item(sheets_unmatched{i}).Delete();
                end
            end

            for i = 1:numel(sheets)
                wb.Sheets.Item(sheets{i}).Name = names{i};
            end
        end

        wb.Save();
        wb.Close();
    catch e
        warning('MATLAB:SystemicRisk',['An error occurred while cleaning the file ''' file '''.' newline() e.message]);
    end
        
    try
        excel.Quit();
        delete(excel);
    catch e1
        warning('MATLAB:SystemicRisk',['An error occurred while disposing the file ''' file ''' (step 1).' newline() e1.message]);

        try
            delete(excel);
        catch e2
            warning('MATLAB:SystemicRisk',['An error occurred while disposing the file ''' file ''' (step 2).' newline() e2.message]);
        end
    end

end

function [sheets,names] = validate_input(sheets,names)

    if (numel(sheets) ~= numel(names))
        error('The ''sheets'' parameter and the ''names'' parameter must contain the same number of elements.');
    end

    if (any(cellfun(@(x)~ischar(x)||isempty(x),sheets)))
        error('The ''sheets'' parameter contains invalid elements.');
    end

    if (~isempty(names) && any(cellfun(@(x)~ischar(x)||isempty(x)||(length(x) > 31),names)))
        error('The ''names'' parameter contains invalid elements.');
    end

end
