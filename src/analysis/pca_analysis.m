%% PCA analysis for wind turbines - ADAML course project

clear; clc; close all;

fprintf('=== ADAML Wind Turbine PCA Analysis ===\n');
fprintf('Team A4 - Lazy Geniuses\n\n');

%% load data and put everything together
fprintf('=== Part 1: Data Loading & Initial Analysis ===\n\n');

% get the turbine data
addpath('../data_loading');
try
    [healthy_data, faulty1_data, faulty2_data, info] = load_turbine_data();
    fprintf('data loaded ok\n');
catch ME
    fprintf('something went wrong loading data: %s\n', ME.message);
    return;
end

% put all data together in one big matrix for pca
fprintf('combining data...\n');
all_data = [healthy_data; faulty1_data; faulty2_data];
[n_obs, n_vars] = size(all_data);
fprintf('total dataset: %d observations x %d variables\n', n_obs, n_vars);

% keep track of which rows belong to which turbine
n_healthy = size(healthy_data, 1);
n_faulty1 = size(faulty1_data, 1);
n_faulty2 = size(faulty2_data, 1);

% make labels for plotting later
labels = [ones(n_healthy, 1);        % healthy = 1
          2*ones(n_faulty1, 1);      % faulty1 = 2
          3*ones(n_faulty2, 1)];     % faulty2 = 3

fprintf('  healthy: %d rows\n', n_healthy);
fprintf('  faulty1: %d rows\n', n_faulty1);
fprintf('  faulty2: %d rows\n', n_faulty2);

% check for missing data
missing_count = sum(isnan(all_data(:)));
if missing_count > 0
    fprintf('found %d missing values - might cause issues\n', missing_count);
else
    fprintf('no missing data - good!\n');
end

%% see which variables are related to each other
fprintf('\n=== checking correlations ===\n');
[corr_matrix, high_corr_pairs] = correlation_analysis(all_data);

fprintf('found %d pairs with high correlation (>0.8)\n', size(high_corr_pairs, 1));
if size(high_corr_pairs, 1) > 0
    fprintf('strongest correlations:\n');
    for i = 1:min(5, size(high_corr_pairs, 1))
        fprintf('  var%d and var%d: r = %.3f\n', ...
                high_corr_pairs(i,1), high_corr_pairs(i,2), high_corr_pairs(i,3));
    end
end

%% basic correlation plot - part of initial data exploration
fprintf('\ncreating initial correlation overview...\n');
figure('Position', [50, 50, 600, 500]);
imagesc(corr_matrix);
colorbar;
colormap('redbluecmap');
clim([-1, 1]);
xlabel('Variable Number');
ylabel('Variable Number');
title('Correlation Matrix - Initial Data Exploration');
grid on;

%% [ARMAN]
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

%% [FASIE - INSERT YOUR VISUALIZATION CODE HERE]
