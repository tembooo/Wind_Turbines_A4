function [corr_matrix, high_corr_pairs] = correlation_analysis(data_matrix)
%% function to check correlations between sensors
% see which sensors move together

fprintf('calculating correlations for %d variables...\n', size(data_matrix, 2));

%% calculate correlation matrix
corr_matrix = corrcoef(data_matrix);

%% find pairs that are highly correlated
% ignore diagonal (variable with itself = 1.0)
cutoff = 0.8;
high_corr_pairs = [];

for i = 1:size(corr_matrix, 1)
    for j = i+1:size(corr_matrix, 2)  % upper triangle only
        if abs(corr_matrix(i,j)) > cutoff
            high_corr_pairs = [high_corr_pairs; i, j, corr_matrix(i,j)];
        end
    end
end

% sort by correlation strength
if ~isempty(high_corr_pairs)
    [~, idx] = sort(abs(high_corr_pairs(:,3)), 'descend');
    high_corr_pairs = high_corr_pairs(idx, :);
end

%% create correlation heatmap visualization
figure('Position', [200, 200, 800, 700]);
imagesc(corr_matrix);
colorbar;
colormap('redbluecmap');  % red=positive, blue=negative
clim([-1, 1]);  % set color limits from -1 to 1

% add labels and formatting
xlabel('variable number');
ylabel('variable number');
title(sprintf('correlation matrix heatmap (%dx%d variables)', size(corr_matrix, 1), size(corr_matrix, 2)));

% add grid lines to separate variables
hold on;
for i = 0.5:1:size(corr_matrix, 1)+0.5
    plot([i, i], [0.5, size(corr_matrix, 1)+0.5], 'k-', 'linewidth', 0.5);
    plot([0.5, size(corr_matrix, 1)+0.5], [i, i], 'k-', 'linewidth', 0.5);
end
hold off;

% add text annotations for very high correlations (> 0.9)
hold on;
for i = 1:size(corr_matrix, 1)
    for j = 1:size(corr_matrix, 2)
        if abs(corr_matrix(i,j)) > 0.9 && i ~= j  % exclude diagonal
            text(j, i, sprintf('%.2f', corr_matrix(i,j)), ...
                 'horizontalalignment', 'center', 'verticalalignment', 'middle', ...
                 'fontsize', 8, 'fontweight', 'bold', 'color', 'white');
        end
    end
end
hold off;

% summary statistics
fprintf('correlation analysis results:\n');
fprintf('  strongest positive correlation: %.3f\n', max(corr_matrix(corr_matrix < 1)));
fprintf('  strongest negative correlation: %.3f\n', min(corr_matrix(:)));
fprintf('  average absolute correlation: %.3f\n', mean(abs(corr_matrix(corr_matrix < 1))));

% distribution of correlations
figure('Position', [250, 250, 600, 400]);
corr_values = corr_matrix(triu(true(size(corr_matrix)), 1));  % upper triangle only
histogram(corr_values, 30, 'facecolor', [0.3, 0.7, 0.9], 'edgecolor', 'black');
xlabel('correlation coefficient');
ylabel('frequency');
title('distribution of pairwise correlations');
grid on; grid minor;

% add vertical lines for strong correlations
hold on;
ylims = ylim;
plot([0.8, 0.8], ylims, 'r--', 'linewidth', 2);
plot([-0.8, -0.8], ylims, 'r--', 'linewidth', 2);
plot([0, 0], ylims, 'k--', 'linewidth', 1);
text(0.8, ylims(2)*0.9, 'strong positive (0.8)', 'rotation', 90, 'horizontalalignment', 'right');
text(-0.8, ylims(2)*0.9, 'strong negative (-0.8)', 'rotation', 90, 'horizontalalignment', 'right');
text(0.05, ylims(2)*0.5, 'no correlation', 'rotation', 90);
hold off;

fprintf('correlation heatmap and distribution plots created!\n');

end