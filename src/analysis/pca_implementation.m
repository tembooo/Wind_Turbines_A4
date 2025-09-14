function [coeffs, scores, latent, explained] = pca_implementation(data_matrix)
%% function to do PCA on our data
% preprocesses the data and runs PCA

fprintf('running PCA on %d observations and %d variables...\n', ...
        size(data_matrix, 1), size(data_matrix, 2));

%% clean up the data first
fprintf('checking data quality...\n');

% look for problems in the data
nan_count = sum(isnan(data_matrix(:)));
inf_count = sum(isinf(data_matrix(:)));
fprintf('  missing values: %d\n', nan_count);
fprintf('  infinite values: %d\n', inf_count);

% handle missing values
if nan_count > 0
    fprintf('  handling missing values by removing affected rows...\n');
    % find rows with any NaN values
    nan_rows = any(isnan(data_matrix), 2);
    fprintf('  removing %d rows with missing values\n', sum(nan_rows));
    data_matrix = data_matrix(~nan_rows, :);
    fprintf('  new data size: %d x %d\n', size(data_matrix));
end

% check for constant columns (zero variance)
var_stds = std(data_matrix);
zero_var_cols = sum(var_stds == 0);
fprintf('  zero variance columns: %d\n', zero_var_cols);

if zero_var_cols > 0
    fprintf('  warning: removing %d constant columns\n', zero_var_cols);
    valid_cols = var_stds > 0;
    data_matrix = data_matrix(:, valid_cols);
    fprintf('  new data size: %d x %d\n', size(data_matrix));
end

% center the data (subtract mean) - pca needs centered data
data_centered = data_matrix - mean(data_matrix);

% check variable scales
var_means = mean(data_matrix);
var_stds = std(data_matrix);

fprintf('variable scale analysis:\n');
fprintf('  mean range: %.3f to %.3f\n', min(var_means), max(var_means));
fprintf('  std range: %.3f to %.3f\n', min(var_stds), max(var_stds));

% if variables have very different scales, we should standardize
scale_ratio = max(var_stds) / min(var_stds);
fprintf('  scale ratio (max/min std): %.2f\n', scale_ratio);

if scale_ratio > 10
    fprintf('large scale differences detected - standardizing data\n');
    data_preprocessed = zscore(data_matrix);  % standardize (mean=0, std=1)
else
    fprintf('similar scales - using centered data only\n');
    data_preprocessed = data_centered;
end

% final check - make sure preprocessed data is valid
fprintf('preprocessed data check:\n');
fprintf('  size: %d x %d\n', size(data_preprocessed));

% check if data is valid for rank calculation
if any(isnan(data_preprocessed(:))) || any(isinf(data_preprocessed(:)))
    fprintf('  warning: preprocessed data contains NaN/Inf values!\n');
    % remove any remaining NaN/Inf
    data_preprocessed(isnan(data_preprocessed)) = 0;
    data_preprocessed(isinf(data_preprocessed)) = 0;
end

try
    data_rank = rank(data_preprocessed);
    fprintf('  rank: %d\n', data_rank);
    fprintf('  condition number: %.2e\n', cond(data_preprocessed));
catch
    fprintf('  warning: could not calculate rank - using SVD approach\n');
    [~, S, ~] = svd(data_preprocessed, 'econ');
    singular_vals = diag(S);
    data_rank = sum(singular_vals > 1e-10);
    fprintf('  effective rank: %d\n', data_rank);
end

%% apply pca using matlab built-in function
fprintf('running pca...\n');
try
    % try the full pca function first
    [coeffs, scores, latent] = pca(data_preprocessed);
    fprintf('pca function completed successfully\n');
    fprintf('  coeffs size: %d x %d\n', size(coeffs));
    fprintf('  scores size: %d x %d\n', size(scores));
    fprintf('  latent length: %d\n', length(latent));

    % calculate explained variance manually from eigenvalues
    explained = 100 * latent / sum(latent);
    fprintf('  calculated explained variance manually\n');
    fprintf('  explained length: %d\n', length(explained));

catch ME
    fprintf('error in pca function: %s\n', ME.message);
    error('pca failed: %s', ME.message);
end

%% analyze results
n_components = length(latent);
fprintf('pca completed successfully!\n');
fprintf('  total components: %d\n', n_components);
fprintf('  total variance: %.2f\n', sum(latent));

% kaiser rule: keep components with eigenvalue > 1
kaiser_components = sum(latent > 1);
fprintf('  kaiser rule suggests keeping %d components\n', kaiser_components);

% scree test: look for elbow in eigenvalue plot
% (we'll let the user see this visually in the main script)

%% component interpretation helpers
fprintf('top contributing variables for first 3 components:\n');

for pc = 1:min(3, n_components)
    [~, top_vars] = sort(abs(coeffs(:, pc)), 'descend');
    fprintf('  pc%d (%.1f%% variance):', pc, explained(pc));
    for i = 1:3  % top 3 contributors
        var_idx = top_vars(i);
        fprintf(' var%d(%.2f)', var_idx, coeffs(var_idx, pc));
    end
    fprintf('\n');
end

%% quality measures
% how well does pca represent the original data?
cumulative_var = cumsum(explained);
pc50 = find(cumulative_var >= 50, 1, 'first');
pc80 = find(cumulative_var >= 80, 1, 'first');
pc95 = find(cumulative_var >= 95, 1, 'first');

fprintf('dimensionality reduction summary:\n');
fprintf('  50%% variance captured by %d components (%.1f%% reduction)\n', ...
        pc50, (1 - pc50/n_components)*100);
fprintf('  80%% variance captured by %d components (%.1f%% reduction)\n', ...
        pc80, (1 - pc80/n_components)*100);
fprintf('  95%% variance captured by %d components (%.1f%% reduction)\n', ...
        pc95, (1 - pc95/n_components)*100);

fprintf('pca implementation completed!\n');

end