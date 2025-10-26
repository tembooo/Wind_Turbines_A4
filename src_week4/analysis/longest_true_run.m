%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Arman Golbidi
% Utility: finds the start/end indices of the longest run of TRUE values 
% in a logical vector (used for contiguous clean windows).

function [startIdx, endIdx] = longest_true_run(good)
% Return start/end indices of the longest run of TRUE in a logical vector

    d = diff([false; good; false]);
    starts = find(d == 1);
    ends   = find(d == -1) - 1;
    if isempty(starts)
        startIdx = []; endIdx = [];
        return;
    end
    [~, k] = max(ends - starts + 1);
    startIdx = starts(k);
    endIdx   = ends(k);
end