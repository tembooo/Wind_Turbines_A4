%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Arman Golbidi
% Helper function for time_aware_preprocess: fills short NaN runs 
% (≤ MaxGap) with linear interpolation and nearest boundary values.


function X_imp = impute_short_gaps(X, MaxGap)
% Fill short NaN runs per column by linear interpolation (ends by nearest)

    X_imp = X;
    if any(isnan(X_imp(:)))
        try
            X_imp = fillmissing(X_imp, 'linear', 'MaxGap', MaxGap, 'EndValues', 'nearest');
        catch
            % Fallback for old MATLAB: manual per-column
            for j = 1:size(X_imp,2)
                x = X_imp(:,j);
                isn = isnan(x);
                if any(isn)
                    t  = (1:numel(x))';
                    xi = interp1(t(~isn), x(~isn), t(isn), 'linear', 'extrap');
                    x(isn) = xi;
                    % Bound ends by nearest valid sample
                    firstValid = find(~isn,1,'first');
                    lastValid  = find(~isn,1,'last');
                    if ~isempty(firstValid), x(1:firstValid-1) = x(firstValid); end
                    if ~isempty(lastValid),  x(lastValid+1:end) = x(lastValid); end
                    X_imp(:,j) = x;
                end
            end
        end
    end
end
