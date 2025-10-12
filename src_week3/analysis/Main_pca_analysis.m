%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Arman Golbidi
% applied time-aware preprocessing (interpolation + longest NaN-free
% window), combined them into one dataset, created labels, checked for
% missing values, and prepared variable names. Initial
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

% Report dropped columns
fprintf('Zero-variance-in-healthy columns dropped: %d\n', numel(scale_info.dropped_idx));
for j = 1:numel(scale_info.dropped_idx)
    k = scale_info.dropped_idx(j);  % Use parentheses for numeric array
    fprintf('  drop col %d (%s), sd_h=%.3g\n', k, scale_info.dropped_names{j}, scale_info.sd_h_full(k));
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
% plots, PC1 vs PC2 scatter, loading plots, biplots, and advanced
% interpretation (separation metrics). His work provides the final
% visualization, interpretation, and reporting of PCA results.

%% === Visualizing Pretreated Data (0.5p) ===
% A) Boxplots, B) Scaled Boxplots, C) Heatmap, D) Timeseries, E) Sanity
% (full code for visualization and advanced interpretation continues here...)

[~, pc2_top] = sort(abs(coeffs(:, 2)), 'descend');
fprintf('\nPC2 (%.1f%% variance) - main contributors:\n', var_explained(2));
for i = 1:min(5, n_vars_kept)
    var_idx = pc2_top(i);
    fprintf('  %s: loading = %.3f\n', var_labels_kept{var_idx}, coeffs(var_idx, 2));
end

%% === Visualizing Pretreated Data (0.5p) ===

% 1) Rebuild the scaled matrix exactly as used by PCA (healthy-based z-scores)
X_raw_kept = all_data(:, scale_info.kept_idx);
mu_h = scale_info.mu_h_full(scale_info.kept_idx);
sd_h = scale_info.sd_h_full(scale_info.kept_idx);
X_scaled = (X_raw_kept - mu_h) ./ sd_h;   % healthy-based scaling

% 2) Row-index ranges for the three classes
idx_h  = 1:n_healthy;
idx_f1 = n_healthy + (1:n_faulty1);
idx_f2 = n_healthy + n_faulty1 + (1:n_faulty2);

% 3) Variable selection: top-3 contributors to PC1 and top-3 to PC2 (deduplicated)
[~, pc1_idx] = sort(abs(coeffs(:,1)), 'descend');
[~, pc2_idx] = sort(abs(coeffs(:,2)), 'descend');
sel_idx = unique([pc1_idx(1:min(3,n_vars_kept)); pc2_idx(1:min(3,n_vars_kept))], 'stable');
sel_idx = sel_idx(1:min(6, numel(sel_idx)));
sel_names = var_labels_kept(sel_idx);

%% A) Boxplot (RAW, post time-aware) — by class
% Use subplot + boxplot (this avoids the MATLAB warning that appears when
% combining boxplot with tiledlayout).
figure('Position',[60,60,1200,500]);
rows = ceil(numel(sel_idx)/3);
cols = min(3, numel(sel_idx));
for k = 1:numel(sel_idx)
    subplot(rows, cols, k);           % one subplot per selected variable
    j = sel_idx(k);                   % column index of the selected variable
    % Concatenate values from the three classes in one vector
    vals = [X_raw_kept(idx_h,j); X_raw_kept(idx_f1,j); X_raw_kept(idx_f2,j)];
    % Matching group labels for the boxplot
    grp  = [repmat({'Healthy'}, numel(idx_h), 1);
            repmat({'Faulty1'},  numel(idx_f1), 1);
            repmat({'Faulty2'},  numel(idx_f2), 1)];
    % Compact style to reduce visual clutter
    boxplot(vals, grp, 'PlotStyle','compact');
    title(sprintf('%s (RAW)', sel_names{k}));
    xlabel('Class'); ylabel('Value'); grid on;
end
sgtitle('Pretreated RAW (post time-aware) — Boxplots by Class');

%% B) Boxplot (healthy-based scaling) — by class
% Here we use tiledlayout + boxchart (boxchart works well with tiledlayout,
% unlike boxplot which can throw a display warning).
figure('Position',[60,580,1200,500]);
rows = ceil(numel(sel_idx)/3);
cols = min(3, numel(sel_idx));
tiledlayout(rows, cols, 'TileSpacing','compact','Padding','compact');
for k = 1:numel(sel_idx)
    ax = nexttile;                    % create/get the next tile axes
    j = sel_idx(k);
    % Values for the three classes (already scaled: healthy-based z-scores)
    vals = [X_scaled(idx_h,j); X_scaled(idx_f1,j); X_scaled(idx_f2,j)];
    % Categorical group labels with a fixed order
    cats = [repmat({'Healthy'}, numel(idx_h), 1);
            repmat({'Faulty1'},  numel(idx_f1), 1);
            repmat({'Faulty2'},  numel(idx_f2), 1)];
    cats = categorical(cats, {'Healthy','Faulty1','Faulty2'});
    % Draw box charts by class
    boxchart(ax, cats, vals);
    title(ax, sprintf('%s (SCALED)', sel_names{k}));
    xlabel(ax,'Class'); ylabel(ax,'z-score (healthy-based)'); grid(ax,'on');
end
sgtitle('Scaled (healthy-based) — Boxplots by Class');


%% C) Heatmap
figure('Position',[200,120,900,750]);
Cscaled = corrcoef(X_scaled);
imagesc(Cscaled); axis image; colorbar;
try colormap(redbluecmap); clim([-1 1]); catch, colormap(parula); end
title('Correlation Matrix (POST-scaling, kept variables)');
xlabel('Variables'); ylabel('Variables');
if n_vars_kept <= 30
    xticks(1:n_vars_kept); yticks(1:n_vars_kept);
    xticklabels(var_labels_kept); yticklabels(var_labels_kept);
    xtickangle(60);
end
grid on;

%% D)
j = pc1_idx(1); vname = var_labels_kept{j};
figure('Position',[220,200,1100,500]);
tiledlayout(3,1,'TileSpacing','compact','Padding','compact');
nexttile; plot(X_scaled(idx_h,j), 'LineWidth',1.0); grid on;
title(sprintf('Healthy — %s (scaled)', vname)); xlabel('Sample index'); ylabel('z');

nexttile; plot(X_scaled(idx_f1,j), 'LineWidth',1.0); grid on;
title(sprintf('Faulty 1 — %s (scaled)', vname)); xlabel('Sample index'); ylabel('z');

nexttile; plot(X_scaled(idx_f2,j), 'LineWidth',1.0); grid on;
title(sprintf('Faulty 2 — %s (scaled)', vname)); xlabel('Sample index'); ylabel('z');

%% E) Sanity check: Healthy
muH = mean(X_scaled(idx_h,:), 1);
sdH = std( X_scaled(idx_h,:), 0, 1);
fprintf('\n[Sanity] Healthy (post-scaling): median(|mean|)=%.3f, median(sd)=%.3f\n', ...
        median(abs(muH)), median(sdH));
% saveas(gcf, 'timeseries_preview_scaled.png');

%% PART 3: ADVANCED VISUALIZATIONS & INTERPRETATION
fprintf('creating detailed PCA visualizations...\n');
% Main overview figure with 4 subplots
figure('Position', [50, 50, 800, 600]);
% Scree plot - eigenvalues
subplot(2, 2, 1);
plot(1:length(eigenvals), eigenvals, 'bo-', 'linewidth', 2, 'markersize', 8);
xlabel('Principal Component Number');
ylabel('Eigenvalue (Variance)');
title('Scree Plot - Eigenvalues');
grid on; grid minor;

% Add Kaiser rule line (eigenvalue = 1)
hold on;
xlims = xlim;
plot(xlims, [1, 1], 'r--', 'linewidth', 2);
text(xlims(2)*0.7, 1.1, 'Kaiser Rule (eigenvalue = 1)', 'color', 'red');
hold off;

% Cumulative variance plot
subplot(2, 2, 2);
plot(1:length(var_explained), cum_var, 'ro-', 'linewidth', 2, 'markersize', 8);
xlabel('Number of Principal Components');
ylabel('Cumulative Variance Explained (%)');
title('Cumulative Variance Plot');
grid on; grid minor;

% Add horizontal lines for 80%, 90%, 95%
hold on;
xlims = xlim;
plot(xlims, [80, 80], 'g--', 'linewidth', 1);
plot(xlims, [90, 90], 'b--', 'linewidth', 1);
plot(xlims, [95, 95], 'm--', 'linewidth', 1);
text(xlims(2)*0.8, 82, '80%', 'color', 'green');
text(xlims(2)*0.8, 92, '90%', 'color', 'blue');
text(xlims(2)*0.8, 97, '95%', 'color', 'magenta');
hold off;

% Scatter plot of PC1 vs PC2 - group separation
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

%% === NEW: PC1–PC2 scores with time/cycle color gradient (per unit) ===
% Put this block right after the class-colored PC Scores subplot (after "hold off;")
% and BEFORE the biplot figure.
% ...
grid on; grid minor;
hold off;
% Keep handle to the overview (2x2) figure to return later
fig_overview = gcf;

%% === NEW: PC1–PC2 scores with time/cycle color gradient (per unit) ===
% Put this block right after the class-colored PC Scores subplot (after "hold off;")
% and BEFORE the biplot figure.

fprintf('\n=== Time-gradient score plots (per turbine) ===\n');

% Build row-index ranges for each turbine block in concatenated data
% (uses n_healthy, n_faulty1, n_faulty2 already computed above)
idx_h  = 1:n_healthy;
idx_f1 = n_healthy + (1:n_faulty1);
idx_f2 = n_healthy + n_faulty1 + (1:n_faulty2);

% 1x3 layout: Healthy | Faulty 1 | Faulty 2
figure('Position',[80,80,1200,380]);
try
    % Prefer tiledlayout if available (R2019b+)
    tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
    nexttile; plot_time_gradient_scores(scores, idx_h,  var_explained, 'Healthy');   % English comments inside the function
    nexttile; plot_time_gradient_scores(scores, idx_f1, var_explained, 'Faulty 1');
    nexttile; plot_time_gradient_scores(scores, idx_f2, var_explained, 'Faulty 2');
catch
    % Fallback to subplot if tiledlayout is not available
    subplot(1,3,1); plot_time_gradient_scores(scores, idx_h,  var_explained, 'Healthy');
    subplot(1,3,2); plot_time_gradient_scores(scores, idx_f1, var_explained, 'Faulty 1');
    subplot(1,3,3); plot_time_gradient_scores(scores, idx_f2, var_explained, 'Faulty 2');
end

% Use a perceptual colormap (always available)
colormap(parula(256));


% Return to the overview (2x2) figure before continuing with subplot(2,2,4)
figure(fig_overview);




% Loading plot for PC1 and PC2 (with kept variable labels)
subplot(2, 2, 4);
plot(coeffs(:, 1), coeffs(:, 2), 'ko', 'markersize', 8, 'markerfacecolor', 'cyan');
hold on;

% Add variable labels (every 3rd variable to avoid clutter)
for i = 1:3:n_vars_kept
    text(coeffs(i, 1)*1.1, coeffs(i, 2)*1.1, var_labels_kept{i}, ...
         'fontsize', 9, 'horizontalalignment', 'center');
end

xlabel(sprintf('PC1 Loadings (%.1f%% variance)', var_explained(1)));
ylabel(sprintf('PC2 Loadings (%.1f%% variance)', var_explained(2)));
title('Loading Plot: Variable Contributions');
grid on; grid minor;

% Add reference lines
xlims = xlim;
ylims = ylim;
plot([0, 0], ylims, 'k--', 'linewidth', 0.5);
plot(xlims, [0, 0], 'k--', 'linewidth', 0.5);
hold off;

sgtitle('Wind Turbine PCA Analysis - Overview', 'fontsize', 16);

% Detailed biplot - separate figure (labels must match kept variables)
figure('Position', [100, 100, 1000, 800]);
biplot(coeffs(:, 1:2), 'scores', scores(:, 1:2), 'varlabels', var_labels_kept);
xlabel(sprintf('PC1 (%.1f%% variance)', var_explained(1)));
ylabel(sprintf('PC2 (%.1f%% variance)', var_explained(2)));
title('PCA Biplot: Variables and Observations in PC Space');
grid on; grid minor;

% Loading contributions for first 3 PCs (bar charts with kept variable labels)
figure('Position', [150, 150, 1200, 400]);
for pc = 1:min(3, size(coeffs,2))
    subplot(1, 3, pc);
    bar(1:n_vars_kept, coeffs(:, pc), 'facecolor', [0.2, 0.6, 0.8]);
    xlabel('Variable');
    ylabel(sprintf('PC%d Loading', pc));
    title(sprintf('PC%d Loadings (%.1f%% var)', pc, var_explained(pc)));
    grid on; grid minor;

    % (Optional) label xticks if not too many variables
    if n_vars_kept <= 30
        xticks(1:n_vars_kept); xticklabels(var_labels_kept); xtickangle(60);
    end

    % Highlight top contributing variables
    [~, top_vars] = sort(abs(coeffs(:, pc)), 'descend');
    for i = 1:min(3, n_vars_kept)  % top 3 contributors
        var_idx = top_vars(i);
        text(var_idx, coeffs(var_idx, pc)*1.1, var_labels_kept{var_idx}, ...
             'horizontalalignment', 'center', 'fontsize', 10, 'fontweight', 'bold');
    end
end
sgtitle('Loading Contributions for First 3 Principal Components', 'fontsize', 14);

%% Detailed interpretation & separation metrics on PC1
fprintf('\n=== Advanced Interpretation & Fault Detection Analysis ===\n');

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

healthy_mean = mean(healthy_pc1);
faulty1_mean = mean(faulty1_pc1);
faulty2_mean = mean(faulty2_pc1);

sep_h_f1 = abs(healthy_mean - faulty1_mean);
sep_h_f2 = abs(healthy_mean - faulty2_mean);
sep_f1_f2 = abs(faulty1_mean - faulty2_mean);

fprintf('\nSeparation distances in PC1:\n');
fprintf('  Healthy vs Faulty1: %.2f (absolute score units)\n', sep_h_f1);
fprintf('  Healthy vs Faulty2: %.2f (absolute score units)\n', sep_h_f2);
fprintf('  Faulty1 vs Faulty2: %.2f (absolute score units)\n', sep_f1_f2);










