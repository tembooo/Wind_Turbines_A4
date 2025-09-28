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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Haider Ali
% He did this part: Visualization and interpretation
% Haider generated pretreated/scaled boxplots, heatmaps, scree & variance
% plots, PC1 vs PC2 scatter, loading plots, biplots, and advanced
% interpretation (separation metrics). His work provides the final
% visualization, interpretation, and reporting of PCA results.











