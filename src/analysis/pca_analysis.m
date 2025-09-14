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
fprintf('\n=== doing PCA ===\n');
[coeffs, scores, eigenvals, var_explained] = pca_implementation(all_data);

% fix labels to match cleaned data (PCA might have removed some rows)
if size(scores, 1) ~= length(labels)
    fprintf('adjusting labels to match cleaned data...\n');
    n_scores = size(scores, 1);
    if n_scores == length(labels) - 1
        % probably removed 1 row with missing data
        % recreate labels proportionally
        ratio_healthy = n_healthy / (n_healthy + n_faulty1 + n_faulty2);
        ratio_faulty1 = n_faulty1 / (n_healthy + n_faulty1 + n_faulty2);

        new_healthy = round(n_scores * ratio_healthy);
        new_faulty1 = round(n_scores * ratio_faulty1);
        new_faulty2 = n_scores - new_healthy - new_faulty1;

        labels = [ones(new_healthy, 1);           % healthy = 1
                  2*ones(new_faulty1, 1);         % faulty1 = 2
                  3*ones(new_faulty2, 1)];        % faulty2 = 3
        fprintf('  new labels: %d healthy, %d faulty1, %d faulty2\n', ...
                new_healthy, new_faulty1, new_faulty2);
    end
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

% how many components needed for different variance levels?
cum_var = cumsum(var_explained);
pc_80 = find(cum_var >= 80, 1, 'first');
pc_90 = find(cum_var >= 90, 1, 'first');
pc_95 = find(cum_var >= 95, 1, 'first');

fprintf('  need %d components for 80%% variance\n', pc_80);
fprintf('  need %d components for 90%% variance\n', pc_90);
fprintf('  need %d components for 95%% variance\n', pc_95);

% basic component interpretation
fprintf('\nlooking at what the main components represent:\n');

% analyze PC1
[~, pc1_top] = sort(abs(coeffs(:, 1)), 'descend');
fprintf('PC1 (%.1f%% variance) - main contributors:\n', var_explained(1));
for i = 1:5
    var_idx = pc1_top(i);
    fprintf('  variable %d: loading = %.3f\n', var_idx, coeffs(var_idx, 1));
end

% analyze PC2
[~, pc2_top] = sort(abs(coeffs(:, 2)), 'descend');
fprintf('\nPC2 (%.1f%% variance) - main contributors:\n', var_explained(2));
for i = 1:5
    var_idx = pc2_top(i);
    fprintf('  variable %d: loading = %.3f\n', var_idx, coeffs(var_idx, 2));
end


%% PART 3: ADVANCED VISUALIZATIONS & INTERPRETATION - Fasie Haider

%% create comprehensive visualizations
fprintf('creating detailed PCA visualizations...\n');

% main overview figure with 4 subplots
figure('Position', [50, 50, 800, 600]);

% scree plot - shows eigenvalues
subplot(2, 2, 1);
plot(1:length(eigenvals), eigenvals, 'bo-', 'linewidth', 2, 'markersize', 8);
xlabel('Principal Component Number');
ylabel('Eigenvalue (Variance)');
title('Scree Plot - Eigenvalues');
grid on; grid minor;

% add kaiser rule line (eigenvalue = 1)
hold on;
xlims = xlim;
plot(xlims, [1, 1], 'r--', 'linewidth', 2);
text(xlims(2)*0.7, 1.1, 'Kaiser Rule (eigenvalue = 1)', 'color', 'red');
hold off;

% cumulative variance plot
subplot(2, 2, 2);
plot(1:length(var_explained), cum_var, 'ro-', 'linewidth', 2, 'markersize', 8);
xlabel('Number of Principal Components');
ylabel('Cumulative Variance Explained (%)');
title('Cumulative Variance Plot');
grid on; grid minor;

% add horizontal lines for 80%, 90%, 95%
hold on;
xlims = xlim;
plot(xlims, [80, 80], 'g--', 'linewidth', 1);
plot(xlims, [90, 90], 'b--', 'linewidth', 1);
plot(xlims, [95, 95], 'm--', 'linewidth', 1);
text(xlims(2)*0.8, 82, '80%', 'color', 'green');
text(xlims(2)*0.8, 92, '90%', 'color', 'blue');
text(xlims(2)*0.8, 97, '95%', 'color', 'magenta');
hold off;

% scatter plot of PC1 vs PC2 - see if turbines separate
subplot(2, 2, 3);
hold on;
colors = ['b', 'r', 'm'];  % blue=healthy, red=faulty1, magenta=faulty2
names = {'healthy', 'faulty1', 'faulty2'};

for t = 1:3
    idx = labels == t;
    scatter(scores(idx, 1), scores(idx, 2), 50, colors(t), 'filled', ...
            'displayname', names{t}, 'markerfacealpha', 0.7);
end

xlabel(sprintf('PC1 (%.1f%% variance)', var_explained(1)));
ylabel(sprintf('PC2 (%.1f%% variance)', var_explained(2)));
title('PC Scores Plot');
legend('location', 'best');
grid on; grid minor;
hold off;

% loading plot for PC1 and PC2
subplot(2, 2, 4);
plot(coeffs(:, 1), coeffs(:, 2), 'ko', 'markersize', 8, 'markerfacecolor', 'cyan');
hold on;

% add variable labels (every 3rd variable to avoid clutter)
for i = 1:3:n_vars
    text(coeffs(i, 1)*1.1, coeffs(i, 2)*1.1, sprintf('v%d', i), ...
         'fontsize', 9, 'horizontalalignment', 'center');
end

xlabel(sprintf('PC1 Loadings (%.1f%% variance)', var_explained(1)));
ylabel(sprintf('PC2 Loadings (%.1f%% variance)', var_explained(2)));
title('Loading Plot: Variable Contributions');
grid on; grid minor;

% add reference lines
xlims = xlim;
ylims = ylim;
plot([0, 0], ylims, 'k--', 'linewidth', 0.5);
plot(xlims, [0, 0], 'k--', 'linewidth', 0.5);
hold off;

sgtitle('Wind Turbine PCA Analysis - Overview', 'fontsize', 16);

% detailed biplot - separate figure
figure('Position', [100, 100, 1000, 800]);
biplot(coeffs(:, 1:2), 'scores', scores(:, 1:2), 'varlabels', ...
       arrayfun(@(x) sprintf('var%d', x), 1:n_vars, 'uniformoutput', false));
xlabel(sprintf('PC1 (%.1f%% variance)', var_explained(1)));
ylabel(sprintf('PC2 (%.1f%% variance)', var_explained(2)));
title('PCA Biplot: Variables and Observations in PC Space');
grid on; grid minor;

% loading contributions for first 3 pcs
figure('Position', [150, 150, 1200, 400]);
for pc = 1:3
    subplot(1, 3, pc);
    bar(1:n_vars, coeffs(:, pc), 'facecolor', [0.2, 0.6, 0.8]);
    xlabel('Variable Number');
    ylabel(sprintf('PC%d Loading', pc));
    title(sprintf('PC%d Loadings (%.1f%% var)', pc, var_explained(pc)));
    grid on; grid minor;

    % highlight top contributing variables
    [~, top_vars] = sort(abs(coeffs(:, pc)), 'descend');
    for i = 1:3  % top 3 contributors
        var_idx = top_vars(i);
        text(var_idx, coeffs(var_idx, pc)*1.1, sprintf('v%d', var_idx), ...
             'horizontalalignment', 'center', 'fontsize', 10, 'fontweight', 'bold');
    end
end
sgtitle('Loading Contributions for First 3 Principal Components', 'fontsize', 14);

%% detailed interpretation and fault detection analysis
fprintf('\n=== Advanced Interpretation & Fault Detection Analysis ===\n');

% turbine separation analysis
fprintf('\nDetailed turbine separation in PC space:\n');
healthy_pc1 = scores(labels == 1, 1);
faulty1_pc1 = scores(labels == 2, 1);
faulty2_pc1 = scores(labels == 3, 1);

fprintf('PC1 statistical analysis:\n');
fprintf('  Healthy: mean=%.2f, std=%.2f, range=[%.2f, %.2f]\n', ...
        mean(healthy_pc1), std(healthy_pc1), min(healthy_pc1), max(healthy_pc1));
fprintf('  Faulty1: mean=%.2f, std=%.2f, range=[%.2f, %.2f]\n', ...
        mean(faulty1_pc1), std(faulty1_pc1), min(faulty1_pc1), max(faulty1_pc1));
fprintf('  Faulty2: mean=%.2f, std=%.2f, range=[%.2f, %.2f]\n', ...
        mean(faulty2_pc1), std(faulty2_pc1), min(faulty2_pc1), max(faulty2_pc1));

% calculate separation metrics
healthy_mean = mean(healthy_pc1);
faulty1_mean = mean(faulty1_pc1);
faulty2_mean = mean(faulty2_pc1);

sep_h_f1 = abs(healthy_mean - faulty1_mean);
sep_h_f2 = abs(healthy_mean - faulty2_mean);
sep_f1_f2 = abs(faulty1_mean - faulty2_mean);

fprintf('\nSeparation distances in PC1:\n');
fprintf('  Healthy vs Faulty1: %.2f standard deviations\n', sep_h_f1);
fprintf('  Healthy vs Faulty2: %.2f standard deviations\n', sep_h_f2);
fprintf('  Faulty1 vs Faulty2: %.2f standard deviations\n', sep_f1_f2);