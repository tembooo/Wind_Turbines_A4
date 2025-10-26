%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Arman Golbidi
% Utility function to detect and drop numeric header rows at the top of sheets.



function [X, dropped] = drop_numeric_header_row(X, sheetname)
% If the first row is a numeric label/header row, drop it.
% Heuristics:
%   - Entire row is finite and integer-valued, AND
%   - (distinct value count <= 5)  OR  (row equals 1:nVars)

    dropped = false;
    if isempty(X), return; end

    r1 = X(1, :);
    is_int_row = all(isfinite(r1)) && all(abs(r1 - round(r1)) < 1e-12);
    looks_like_index = isequal(r1, 1:size(X, 2));
    few_distinct_ints = numel(unique(r1)) <= 5;

    if is_int_row && (few_distinct_ints || looks_like_index)
        X(1, :) = [];
        dropped = true;
        fprintf('  removed numeric header/label ROW from sheet %s\n', sheetname);
    end
end
