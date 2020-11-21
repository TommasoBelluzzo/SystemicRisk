% [OUTPUT]
% c = A newline character.

%#ok<*SPRINTFN>
function c = new_line()

    if (verLessThan('MATLAB','9.1'))
        c = sprintf('\n');
    else
        c = newline();
    end

end
