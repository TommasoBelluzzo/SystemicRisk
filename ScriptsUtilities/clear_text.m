% [INPUT]
% txt = A string representing the text to clear.
%
% [OUTPUT]
% out = A string representing the escaped text.
%
% [NOTES]
% The function strips HTML tags and escapes backslashes.

function out = clear_text(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('txt',@(x)validateattributes(x,{'char'},{}));
    end

    ip.parse(varargin{:});

    ipr = ip.Results;
    txt = ipr.txt;

    nargoutchk(1,1);

    out = clear_text_internal(txt);

end

function out = clear_text_internal(txt)

    if (isempty(txt))
        out = txt;
    else
        out = regexprep(txt,'<[^>]*>','');
        out = strrep(out,filesep(),[filesep() filesep()]);
    end

end
