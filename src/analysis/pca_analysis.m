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

%% [ARMAN - INSERT YOUR PCA IMPLEMENTATION CODE HERE]


%% [FASIE - INSERT YOUR VISUALIZATION CODE HERE]
