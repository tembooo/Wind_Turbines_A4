%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Arman Golbidi
% He did this part: Data loading, pretreatment, initial checks
% Arman implemented the loading of Healthy/Faulty turbine datasets,
% applied time-aware preprocessing (interpolation + longest NaN-free
% window), combined them into one dataset, created labels, checked for
% missing values, and prepared variable names. He also performed the
% first exploratory correlation analysis.

%% PCA analysis for wind turbines - ADAML course project

clear; clc; close all;
addpath('../analysis');       % analysis module
addpath('../data_loading');   % data loading module


fprintf('=== ADAML Wind Turbine PCA Analysis ===\n');
fprintf('Team A4 - Lazy Geniuses\n\n');

%% load data and put everything together
fprintf('=== Part 1: Data Loading & Initial Analysis ===\n\n');

% Keep original addpath (no behavior change even if functions are local)
addpath('../data_loading');

try
    [healthy_data, faulty1_data, faulty2_data, info] = load_turbine_data();
    fprintf('data loaded ok\n');

    %% NEW: Time-aware pretreatment (unit-wise)
    % Short gaps -> interpolate (≤ MaxGap); long gaps -> keep longest complete window
    MaxGap = 3;  % tune if needed (in samples)

    [healthy_data, faulty1_data, faulty2_data, ta_info] = time_aware_preprocess( ...
        healthy_data, faulty1_data, faulty2_data, MaxGap);

    fprintf('kept rows after time-aware preprocessing: healthy=%d, f1=%d, f2=%d\n', ...
        size(healthy_data,1), size(faulty1_data,1), size(faulty2_data,1));

catch ME
    fprintf('something went wrong loading data: %s\n', ME.message);
    return;
end

% Put all data together in one big matrix for PCA
fprintf('combining data...\n');
all_data = [healthy_data; faulty1_data; faulty2_data];
[n_obs, n_vars_raw] = size(all_data);
fprintf('total dataset: %d observations x %d variables\n', n_obs, n_vars_raw);

% Keep track of which rows belong to which turbine
n_healthy = size(healthy_data, 1);
n_faulty1 = size(faulty1_data, 1);
n_faulty2 = size(faulty2_data, 1);

% Labels for plotting later
labels = [ones(n_healthy, 1);        % healthy = 1
          2*ones(n_faulty1, 1);      % faulty1 = 2
          3*ones(n_faulty2, 1)];     % faulty2 = 3

fprintf('  healthy: %d rows\n', n_healthy);
fprintf('  faulty1: %d rows\n', n_faulty1);
fprintf('  faulty2: %d rows\n', n_faulty2);

% Prepare variable names (from Excel if available; otherwise v1..vP)
if isfield(info,'var_names') && ~isempty(info.var_names)
    var_names = info.var_names(:).';          % 1xP cellstr
else
    var_names = arrayfun(@(k) sprintf('v%d',k), 1:size(all_data,2), 'UniformOutput', false);
end

% Basic missing data check (post pretreatment this should be zero)
missing_count = sum(isnan(all_data(:)));
if missing_count > 0
    fprintf('warning: found %d missing values after pretreatment\n', missing_count);
else
    fprintf('no missing data - good!\n');
end

%% See which variables are related to each other (exploratory, unscaled)
fprintf('\n=== checking correlations (exploratory) ===\n');
[corr_matrix, high_corr_pairs] = correlation_analysis(all_data);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Fasie Haider
% He did this part: PCA modelling, scaling, variance explained
% Fasie implemented the PCA with healthy-based scaling, handled
% zero-variance variables, computed explained variance, number of PCs
% for 80/90/95%, and identified the top contributing variables for PC1
% and PC2.
fprintf('found %d pairs with high correlation (>0.8)\n', size(high_corr_pairs, 1));
if size(high_corr_pairs, 1) > 0
    fprintf('strongest correlations:\n');
    for i = 1:min(5, size(high_corr_pairs, 1))
        fprintf('  %s and %s: r = %.3f\n', ...
            var_names{high_corr_pairs(i,1)}, var_names{high_corr_pairs(i,2)}, high_corr_pairs(i,3));
    end
end

%% PCA (with healthy-based scaling inside)
fprintf('\n=== doing PCA (healthy-based scaling) ===\n');
[coeffs, scores, eigenvals, var_explained, scale_info] = ...
    pca_implementation(all_data, n_healthy, var_names);

% Report which columns were dropped (zero variance in healthy)
fprintf('Zero-variance-in-healthy columns dropped: %d\n', numel(scale_info.dropped_idx));
for k = 1:numel(scale_info.dropped_idx)
    j = scale_info.dropped_idx(k);
    fprintf('  drop col %d (%s), sd_h=%.3g\n', j, scale_info.dropped_names{k}, scale_info.sd_h_full(j));
end
fprintf('Columns before: %d | after: %d\n', scale_info.n_vars_in, numel(scale_info.kept_idx));

% Short handles for kept labels
var_labels_kept = scale_info.kept_names;      % 1xP_kept cellstr
n_vars_kept     = numel(var_labels_kept);

% If any unexpected row removal somehow happened (should not), adjust labels defensively
if size(scores, 1) ~= length(labels)
    fprintf('adjusting labels to match cleaned data...\n');
    n_scores = size(scores, 1);
    ratio_healthy = n_healthy / (n_healthy + n_faulty1 + n_faulty2);
    ratio_faulty1 = n_faulty1 / (n_healthy + n_faulty1 + n_faulty2);

    new_healthy = round(n_scores * ratio_healthy);
    new_faulty1 = round(n_scores * ratio_faulty1);
    new_faulty2 = n_scores - new_healthy - new_faulty1;

    labels = [ones(new_healthy, 1);
              2*ones(new_faulty1, 1);
              3*ones(new_faulty2, 1)];
    fprintf('  new labels: %d healthy, %d faulty1, %d faulty2\n', ...
            new_healthy, new_faulty1, new_faulty2);
end

fprintf('PCA results:\n');
if isempty(var_explained) || length(var_explained) < 1
    fprintf('error: PCA failed!\n');
    return;
end

fprintf('  PC1 explains: %.1f%% of variance\n', var_explained(1));
if length(var_explained) >= 2
    fprintf('  PC1+PC2 explain: %.1f%% total\n', sum(var_explained(1:2)));
end
if length(var_explained) >= 5
    fprintf('  first 5 PCs explain: %.1f%% total\n', sum(var_explained(1:5)));
end

% How many components needed for different variance levels?
cum_var = cumsum(var_explained);
pc_80 = find(cum_var >= 80, 1, 'first');
pc_90 = find(cum_var >= 90, 1, 'first');
pc_95 = find(cum_var >= 95, 1, 'first');

fprintf('  need %d components for 80%% variance\n', pc_80);
fprintf('  need %d components for 90%% variance\n', pc_90);
fprintf('  need %d components for 95%% variance\n', pc_95);
% Basic component interpretation (print top contributors by NAME)
fprintf('\nlooking at what the main components represent:\n');
[~, pc1_top] = sort(abs(coeffs(:, 1)), 'descend');
fprintf('PC1 (%.1f%% variance) - main contributors:\n', var_explained(1));
for i = 1:min(5, n_vars_kept)
    var_idx = pc1_top(i);
    fprintf('  %s: loading = %.3f\n', var_labels_kept{var_idx}, coeffs(var_idx, 1));
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Haider Ali
% He did this part: Visualization and interpretation
% Haider generated pretreated/scaled boxplots, heatmaps, scree & variance
% plots, PC1 vs PC2 scatter, loading plots, biplots, and advanced
% interpretation (separation metrics). His work provides the final
% visualization, interpretation, and reporting of PCA results.











