%%%%%%%%%%%%%%% 0.5p – Visualization & Comments (Arman Golbidi)
% Summary statistics
min_vals   = min(allData);
max_vals   = max(allData);
means      = mean(allData);
medians    = median(allData);
std_devs   = std(allData);

statsTable = table(min_vals', max_vals', means', medians', std_devs', ...
    'VariableNames', {'Min', 'Max', 'Mean', 'Median', 'StdDev'}, ...
    'RowNames', arrayfun(@num2str, (1:numAttributes)', 'UniformOutput', false));
disp('Summary Statistics Table:');
disp(statsTable);

% Histograms for first variables
figure;
for i = 1:min(numAttributes, 5)
    subplot(2, 3, i);
    histogram(allData(:, i));
    title(['Variable ', num2str(i)]);
end

% Correlation heatmap
correlationMatrix = corr(allData);
figure;
heatmap(correlationMatrix);
colorbar;
title('Correlation Matrix Heatmap');


%%%%%%%%%%%%%%% PCA – Arman Golbidi
% Step 4: Explained variance plot
figure;
plot(cumsum(explained), '-o');
title('Explained Variance by Principal Components');
xlabel('Principal Components'); ylabel('Cumulative Variance Explained (%)');
grid on;
