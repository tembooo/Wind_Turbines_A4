%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Arman Golbidi
% Performs time-aware preprocessing: interpolate short gaps, 
% and cut to longest contiguous NaN-free window.


function [Xh_out, X1_out, X2_out, info] = time_aware_preprocess(Xh, X1, X2, MaxGap)
% Unit-wise, time-aware pretreatment:
% - Interpolate short NaN runs (≤ MaxGap) per variable
% - If long NaN runs remain, keep the longest contiguous window with no NaN

    if nargin < 4, MaxGap = 3; end

    % 1) interpolate short gaps
    Xh_imp  = impute_short_gaps(Xh,  MaxGap);
    X1_imp  = impute_short_gaps(X1,  MaxGap);
    X2_imp  = impute_short_gaps(X2,  MaxGap);

    % 2) cut to longest complete window if NaNs still exist
    [Xh_out, s_h,  e_h ] = select_longest_complete_window(Xh_imp);
    [X1_out, s_1,  e_1 ] = select_longest_complete_window(X1_imp);
    [X2_out, s_2,  e_2 ] = select_longest_complete_window(X2_imp);

    fprintf('time-aware: healthy [%d→%d], f1 [%d→%d], f2 [%d→%d] (rows kept)\n', ...
        size(Xh,1), size(Xh_out,1), size(X1,1), size(X1_out,1), size(X2,1), size(X2_out,1));

    info = struct('MaxGap', MaxGap, ...
        'healthy_window', [s_h e_h], 'faulty1_window', [s_1 e_1], 'faulty2_window', [s_2 e_2]);
end
