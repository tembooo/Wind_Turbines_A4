%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Arman Golbidi
% Helper for time_aware_preprocess: keeps the longest contiguous 
% segment with no NaN across all variables.



function [X_win, s, e] = select_longest_complete_window(X)
% Keep the longest contiguous block with no NaN across all variables

    good = all(~isnan(X), 2);
    [s, e] = longest_true_run(good);
    if isempty(s)
        error('No NaN-free contiguous window found. Consider increasing MaxGap or inspecting sensors.');
    end
    X_win = X(s:e, :);
end