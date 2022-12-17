% [OUTPUT]
% c = A newline character.

function c = new_line()

    if (verLessThan('MATLAB','9.1'))
        c = sprintf('\n'); %#ok<SPRINTFN> 
    else
        c = newline();
    end

end
