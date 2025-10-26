%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% [Your Name or Team]
% This function computes correlation matrix and identifies high correlation pairs.
% - Input: data matrix (observations x variables)
% - Output: correlation matrix, pairs with |corr| > threshold

function [corr_matrix, high_corr_pairs] = correlation_analysis(data_matrix)
% Compute correlations for exploratory analysis

    fprintf('calculating correlations for %d variables...\n', size(data_matrix, 2));

    % Compute correlation matrix
    corr_matrix = corrcoef(data_matrix);
    
    % Identify high correlation pairs (e.g., |corr| > 0.7)
    n_vars = size(data_matrix, 2);
    high_corr_pairs = [];
    threshold = 0.7;
    for i = 1:n_vars-1
        for j = i+1:n_vars
            if abs(corr_matrix(i,j)) > threshold
                high_corr_pairs = [high_corr_pairs; [i j corr_matrix(i,j)]];
            end
        end
    end

    % Visualize correlation matrix (heatmap)
    figure('Position', [100, 100, 800, 600]);
    imagesc(corr_matrix);
    colorbar;
    title('Correlation Matrix Heatmap');
    xlabel('Variable Index');
    ylabel('Variable Index');
    % Replace redbluecmap with a built-in colormap (e.g., cool or parula)
    colormap('cool');  % Use 'parula', 'jet', or 'cool' instead of 'redbluecmap'
    % Add grid for clarity
    grid on;

    % Display high correlation pairs
    if ~isempty(high_corr_pairs)
        fprintf('High correlation pairs (|corr| > %.2f):\n', threshold);
        for k = 1:size(high_corr_pairs, 1)
            fprintf('Vars %d and %d: corr = %.3f\n', high_corr_pairs(k,1), high_corr_pairs(k,2), high_corr_pairs(k,3));
        end
    end
end
