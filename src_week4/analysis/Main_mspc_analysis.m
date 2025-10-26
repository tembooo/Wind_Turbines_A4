%% MSPC analysis for wind turbines - ADML course project (PCA vs k-PCA)

clear; clc; close all;
addpath('../analysis');       % analysis module
addpath('../data_loading');   % data loading module

% Create figures directory if it doesn't exist
if ~exist('../figures', 'dir')
    mkdir('../figures');
end

fprintf('=== ADML Wind Turbine MSPC Analysis (PCA vs k-PCA) ===\n');
fprintf('Team A4 - Lazy Geniuses\n\n');

%% Load and preprocess (same as Main_pca_analysis)
fprintf('=== Data Loading & Pretreatment ===\n');

[healthy_data, faulty1_data, faulty2_data, info] = load_turbine_data();

MaxGap = 3;
[healthy_data, faulty1_data, faulty2_data, ta_info] = time_aware_preprocess( ...
    healthy_data, faulty1_data, faulty2_data, MaxGap);

all_data = [healthy_data; faulty1_data; faulty2_data];
[n_obs, n_vars_raw] = size(all_data);

n_healthy = size(healthy_data,1);
n_faulty1 = size(faulty1_data,1);
n_faulty2 = size(faulty2_data,1);

var_names = info.var_names;  % Or fallback to v1..v27

alpha_level = 0.05;  % For limits (95%)
sigma = 1.5;  % Tuned kernel width (optimize via CV FAR minimization)

%% Fit PCA (using existing function)
fprintf('\n=== Fitting PCA Model ===\n');
[coeffs, scores_pca, latent, explained_pca, scale_info] = ...
    pca_implementation(all_data, n_healthy, var_names);

a = 4;  % Optimized: PC1-4 ~92% variance (from scree/cumvar/Kaiser)
coeffs = coeffs(:,1:a);
latent = latent(1:a);
scores_pca = scores_pca(:,1:a);

Z = (all_data(:, scale_info.kept_idx) - scale_info.mu_h) ./ scale_info.sd_h;
hat_Z = scores_pca * coeffs';

[T2_pca, SPE_pca, T2_limit_pca, SPE_limit_pca] = ...
    compute_stats('pca', scores_pca, latent, alpha_level, n_healthy, Z, hat_Z, []);

%% Fit k-PCA
fprintf('\n=== Fitting k-PCA Model ===\n');
[alpha, lambda, scores_kpca, explained_kpca, ~, kernel_info] = ...
    kpca_implementation(all_data, n_healthy, var_names, sigma);

alpha = alpha(:,1:a);
lambda = lambda(1:a);
scores_kpca = scores_kpca(:,1:a);

[T2_kpca, SPE_kpca, T2_limit_kpca, SPE_limit_kpca] = ...
    compute_stats('kpca', scores_kpca, lambda, alpha_level, n_healthy, Z, [], kernel_info);

%% === MODEL DIAGNOSTIC PLOTS ===
fprintf('\n=== Generating Model Diagnostic Plots ===\n');

% PCA: Scree plot + Cumulative Variance
figure('Position', [100, 100, 1000, 450]);
subplot(1,2,1);
stem(1:min(15,length(latent)), latent(1:min(15,length(latent))), 'filled', 'LineWidth', 1.5, 'MarkerSize', 8);
hold on;
yline(1, 'r--', 'Kaiser Criterion', 'LineWidth', 1.8, 'FontSize', 12);
xlabel('Principal Component', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('\lambda_k (Eigenvalue)', 'FontSize', 13, 'FontWeight', 'bold');
title('PCA Scree Plot', 'FontSize', 14, 'FontWeight', 'bold');
grid on; set(gca, 'FontSize', 12);
xlim([0.5, min(15,length(latent))+0.5]);

subplot(1,2,2);
cum_var_pca = cumsum(explained_pca);
plot(1:min(15,length(cum_var_pca)), cum_var_pca(1:min(15,length(cum_var_pca))), ...
     'b-o', 'LineWidth', 2.5, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
hold on;
yline(90, 'g--', '90% Variance', 'LineWidth', 1.8, 'FontSize', 12);
yline(95, 'm--', '95% Variance', 'LineWidth', 1.8, 'FontSize', 12);
xlabel('Number of Components', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('Cumulative Variance Explained (%)', 'FontSize', 13, 'FontWeight', 'bold');
title('PCA Cumulative Variance', 'FontSize', 14, 'FontWeight', 'bold');
grid on; ylim([0 105]); set(gca, 'FontSize', 12);
xlim([0.5, min(15,length(cum_var_pca))+0.5]);
saveas(gcf, '../figures/pca_diagnostics_scree_cumvar.png');

% k-PCA: Kernel parameter selection (σ tuning via FAR minimization)
fprintf('Kernel parameter tuning: testing σ in [0.5, 3.0]...\n');
sigma_range = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0];
FAR_kpca_grid = zeros(length(sigma_range), 1);

for s_idx = 1:length(sigma_range)
    sig_test = sigma_range(s_idx);
    [alpha_temp, lambda_temp, scores_temp, ~, ~, ki_temp] = ...
        kpca_implementation(all_data, n_healthy, var_names, sig_test);
    alpha_temp = alpha_temp(:,1:a);
    lambda_temp = lambda_temp(1:a);
    scores_temp = scores_temp(:,1:a);
    [T2_temp, SPE_temp, T2_lim_temp, SPE_lim_temp] = ...
        compute_stats('kpca', scores_temp, lambda_temp, alpha_level, n_healthy, Z, [], ki_temp);
    alarms_h = (T2_temp(1:n_healthy) > T2_lim_temp) | (SPE_temp(1:n_healthy) > SPE_lim_temp);
    FAR_kpca_grid(s_idx) = mean(alarms_h);
    fprintf('  σ=%.2f: FAR=%.3f\n', sig_test, FAR_kpca_grid(s_idx));
end

figure('Position', [120, 120, 800, 500]);
plot(sigma_range, FAR_kpca_grid, 'ro-', 'LineWidth', 2.5, 'MarkerSize', 10, 'MarkerFaceColor', 'r');
xlabel('\sigma (Gaussian kernel width)', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('FAR on Healthy Data', 'FontSize', 13, 'FontWeight', 'bold');
title('k-PCA: Kernel Width Selection via FAR Minimization', 'FontSize', 14, 'FontWeight', 'bold');
grid on; set(gca, 'FontSize', 12);
xline(sigma, 'b--', sprintf('Selected \\sigma = %.2f', sigma), 'LineWidth', 2, 'FontSize', 12);
saveas(gcf, '../figures/kpca_kernel_tuning_FAR.png');

%% Validation on Healthy (contiguous CV)
fprintf('\n=== Model Validation (Healthy CV) ===\n');
[avg_FAR_T2_pca, avg_FAR_SPE_pca, avg_ARL_pca] = ...
    validate_model('pca', all_data, n_healthy, var_names, alpha_level);

[avg_FAR_T2_kpca, avg_FAR_SPE_kpca, avg_ARL_kpca] = ...
    validate_model('kpca', all_data, n_healthy, var_names, alpha_level, sigma);

fprintf('PCA: Avg FAR T2=%.3f, SPE=%.3f; ARL=%.1f\n', ...
        avg_FAR_T2_pca, avg_FAR_SPE_pca, avg_ARL_pca);
fprintf('k-PCA: Avg FAR T2=%.3f, SPE=%.3f; ARL=%.1f\n', ...
        avg_FAR_T2_kpca, avg_FAR_SPE_kpca, avg_ARL_kpca);

%% Testing on Faulty & Detection Results
fprintf('\n=== Testing on Faulty Turbines ===\n');

idx_f1 = n_healthy + (1:n_faulty1);
idx_f2 = n_healthy + n_faulty1 + (1:n_faulty2);

% PCA detection
det_rate_pca_f1 = mean( (T2_pca(idx_f1) > T2_limit_pca) | (SPE_pca(idx_f1) > SPE_limit_pca) );
det_rate_pca_f2 = mean( (T2_pca(idx_f2) > T2_limit_pca) | (SPE_pca(idx_f2) > SPE_limit_pca) );
ttd_pca_f1 = find( (T2_pca(idx_f1) > T2_limit_pca) | (SPE_pca(idx_f1) > SPE_limit_pca), 1 );
ttd_pca_f2 = find( (T2_pca(idx_f2) > T2_limit_pca) | (SPE_pca(idx_f2) > SPE_limit_pca), 1 );

fprintf('PCA Detection: WT14 %.1f%% (ttd=%d), WT39 %.1f%% (ttd=%d)\n', ...
        det_rate_pca_f1*100, ttd_pca_f1, det_rate_pca_f2*100, ttd_pca_f2);

% k-PCA detection
det_rate_kpca_f1 = mean( (T2_kpca(idx_f1) > T2_limit_kpca) | (SPE_kpca(idx_f1) > SPE_limit_kpca) );
det_rate_kpca_f2 = mean( (T2_kpca(idx_f2) > T2_limit_kpca) | (SPE_kpca(idx_f2) > SPE_limit_kpca) );
ttd_kpca_f1 = find( (T2_kpca(idx_f1) > T2_limit_kpca) | (SPE_kpca(idx_f1) > SPE_limit_kpca), 1 );
ttd_kpca_f2 = find( (T2_kpca(idx_f2) > T2_limit_kpca) | (SPE_kpca(idx_f2) > SPE_limit_kpca), 1 );

fprintf('k-PCA Detection: WT14 %.1f%% (ttd=%d), WT39 %.1f%% (ttd=%d)\n', ...
        det_rate_kpca_f1*100, ttd_kpca_f1, det_rate_kpca_f2*100, ttd_kpca_f2);

%% === CONTROL CHARTS: HEALTHY + FAULTY ===
fprintf('\n=== Generating Control Charts (Healthy + Faulty) ===\n');

idx_all_h  = 1:n_healthy;
idx_all_f1 = n_healthy + (1:n_faulty1);
idx_all_f2 = n_healthy + n_faulty1 + (1:n_faulty2);

% --- PCA Control Charts for WT14 (Healthy + Faulty1) ---
figure('Position', [100, 100, 1300, 600]);
subplot(2,1,1);
hold on;
patch([idx_all_h(1) idx_all_h(end) idx_all_h(end) idx_all_h(1)], ...
      [0 0 max(T2_pca)*1.15 max(T2_pca)*1.15], [0.85 0.95 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.4);
plot(idx_all_h, T2_pca(idx_all_h), 'g-', 'LineWidth', 1.5, 'DisplayName', 'WT2 (Healthy)');
plot(idx_all_f1, T2_pca(idx_all_f1), 'b-', 'LineWidth', 1.5, 'DisplayName', 'WT14 (Faulty)');
yline(T2_limit_pca, 'r--', 'Control Limit (\alpha=0.05)', 'LineWidth', 2.2, 'FontSize', 12);
xlabel('Sample Index', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('\bfT^2 Statistic', 'FontSize', 13);
title('PCA Control Chart: \bfT^2 (Healthy WT2 + Faulty WT14)', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 11);
grid on; set(gca, 'FontSize', 12);

subplot(2,1,2);
hold on;
patch([idx_all_h(1) idx_all_h(end) idx_all_h(end) idx_all_h(1)], ...
      [0 0 max(SPE_pca)*1.15 max(SPE_pca)*1.15], [0.85 0.95 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.4);
plot(idx_all_h, SPE_pca(idx_all_h), 'g-', 'LineWidth', 1.5, 'DisplayName', 'WT2 (Healthy)');
plot(idx_all_f1, SPE_pca(idx_all_f1), 'b-', 'LineWidth', 1.5, 'DisplayName', 'WT14 (Faulty)');
yline(SPE_limit_pca, 'r--', 'Control Limit (\alpha=0.05)', 'LineWidth', 2.2, 'FontSize', 12);
xlabel('Sample Index', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('\bfSPE (Q) Statistic', 'FontSize', 13);
title('PCA Control Chart: \bfSPE (Healthy WT2 + Faulty WT14)', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 11);
grid on; set(gca, 'FontSize', 12);
saveas(gcf, '../figures/pca_control_charts_WT14_with_healthy.png');

% --- PCA Control Charts for WT39 ---
figure('Position', [120, 120, 1300, 600]);
subplot(2,1,1);
hold on;
patch([idx_all_h(1) idx_all_h(end) idx_all_h(end) idx_all_h(1)], ...
      [0 0 max(T2_pca)*1.15 max(T2_pca)*1.15], [0.85 0.95 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.4);
plot(idx_all_h, T2_pca(idx_all_h), 'g-', 'LineWidth', 1.5, 'DisplayName', 'WT2 (Healthy)');
plot(idx_all_f2, T2_pca(idx_all_f2), 'm-', 'LineWidth', 1.5, 'DisplayName', 'WT39 (Faulty)');
yline(T2_limit_pca, 'r--', 'Control Limit (\alpha=0.05)', 'LineWidth', 2.2, 'FontSize', 12);
xlabel('Sample Index', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('\bfT^2 Statistic', 'FontSize', 13);
title('PCA Control Chart: \bfT^2 (Healthy WT2 + Faulty WT39)', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 11);
grid on; set(gca, 'FontSize', 12);

subplot(2,1,2);
hold on;
patch([idx_all_h(1) idx_all_h(end) idx_all_h(end) idx_all_h(1)], ...
      [0 0 max(SPE_pca)*1.15 max(SPE_pca)*1.15], [0.85 0.95 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.4);
plot(idx_all_h, SPE_pca(idx_all_h), 'g-', 'LineWidth', 1.5, 'DisplayName', 'WT2 (Healthy)');
plot(idx_all_f2, SPE_pca(idx_all_f2), 'm-', 'LineWidth', 1.5, 'DisplayName', 'WT39 (Faulty)');
yline(SPE_limit_pca, 'r--', 'Control Limit (\alpha=0.05)', 'LineWidth', 2.2, 'FontSize', 12);
xlabel('Sample Index', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('\bfSPE (Q) Statistic', 'FontSize', 13);
title('PCA Control Chart: \bfSPE (Healthy WT2 + Faulty WT39)', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 11);
grid on; set(gca, 'FontSize', 12);
saveas(gcf, '../figures/pca_control_charts_WT39_with_healthy.png');

% --- k-PCA Control Charts for WT14 ---
figure('Position', [140, 140, 1300, 600]);
subplot(2,1,1);
hold on;
patch([idx_all_h(1) idx_all_h(end) idx_all_h(end) idx_all_h(1)], ...
      [0 0 max(T2_kpca)*1.15 max(T2_kpca)*1.15], [0.85 0.95 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.4);
plot(idx_all_h, T2_kpca(idx_all_h), 'g-', 'LineWidth', 1.5, 'DisplayName', 'WT2 (Healthy)');
plot(idx_all_f1, T2_kpca(idx_all_f1), 'b-', 'LineWidth', 1.5, 'DisplayName', 'WT14 (Faulty)');
yline(T2_limit_kpca, 'r--', 'Control Limit (empirical 95%)', 'LineWidth', 2.2, 'FontSize', 12);
xlabel('Sample Index', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('\bfT^2 Statistic', 'FontSize', 13);
title('k-PCA Control Chart: \bfT^2 (Healthy WT2 + Faulty WT14)', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 11);
grid on; set(gca, 'FontSize', 12);

subplot(2,1,2);
hold on;
patch([idx_all_h(1) idx_all_h(end) idx_all_h(end) idx_all_h(1)], ...
      [0 0 max(SPE_kpca)*1.15 max(SPE_kpca)*1.15], [0.85 0.95 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.4);
plot(idx_all_h, SPE_kpca(idx_all_h), 'g-', 'LineWidth', 1.5, 'DisplayName', 'WT2 (Healthy)');
plot(idx_all_f1, SPE_kpca(idx_all_f1), 'b-', 'LineWidth', 1.5, 'DisplayName', 'WT14 (Faulty)');
yline(SPE_limit_kpca, 'r--', 'Control Limit (empirical 95%)', 'LineWidth', 2.2, 'FontSize', 12);
xlabel('Sample Index', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('\bfSPE (Q) Statistic', 'FontSize', 13);
title('k-PCA Control Chart: \bfSPE (Healthy WT2 + Faulty WT14)', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 11);
grid on; set(gca, 'FontSize', 12);
saveas(gcf, '../figures/kpca_control_charts_WT14_with_healthy.png');

% --- k-PCA Control Charts for WT39 ---
figure('Position', [160, 160, 1300, 600]);
subplot(2,1,1);
hold on;
patch([idx_all_h(1) idx_all_h(end) idx_all_h(end) idx_all_h(1)], ...
      [0 0 max(T2_kpca)*1.15 max(T2_kpca)*1.15], [0.85 0.95 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.4);
plot(idx_all_h, T2_kpca(idx_all_h), 'g-', 'LineWidth', 1.5, 'DisplayName', 'WT2 (Healthy)');
plot(idx_all_f2, T2_kpca(idx_all_f2), 'm-', 'LineWidth', 1.5, 'DisplayName', 'WT39 (Faulty)');
yline(T2_limit_kpca, 'r--', 'Control Limit (empirical 95%)', 'LineWidth', 2.2, 'FontSize', 12);
xlabel('Sample Index', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('\bfT^2 Statistic', 'FontSize', 13);
title('k-PCA Control Chart: \bfT^2 (Healthy WT2 + Faulty WT39)', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 11);
grid on; set(gca, 'FontSize', 12);

subplot(2,1,2);
hold on;
patch([idx_all_h(1) idx_all_h(end) idx_all_h(end) idx_all_h(1)], ...
      [0 0 max(SPE_kpca)*1.15 max(SPE_kpca)*1.15], [0.85 0.95 0.85], 'EdgeColor', 'none', 'FaceAlpha', 0.4);
plot(idx_all_h, SPE_kpca(idx_all_h), 'g-', 'LineWidth', 1.5, 'DisplayName', 'WT2 (Healthy)');
plot(idx_all_f2, SPE_kpca(idx_all_f2), 'm-', 'LineWidth', 1.5, 'DisplayName', 'WT39 (Faulty)');
yline(SPE_limit_kpca, 'r--', 'Control Limit (empirical 95%)', 'LineWidth', 2.2, 'FontSize', 12);
xlabel('Sample Index', 'FontSize', 13, 'FontWeight', 'bold');
ylabel('\bfSPE (Q) Statistic', 'FontSize', 13);
title('k-PCA Control Chart: \bfSPE (Healthy WT2 + Faulty WT39)', 'FontSize', 14, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 11);
grid on; set(gca, 'FontSize', 12);
saveas(gcf, '../figures/kpca_control_charts_WT39_with_healthy.png');

%% === CONTRIBUTION ANALYSIS: T² AND SPE for BOTH MODELS ===
fprintf('\n=== Diagnostics: Contributions ===\n');

first_alarm_pca = n_healthy + ttd_pca_f1;
first_alarm_kpca = n_healthy + ttd_kpca_f1;

% PCA Contributions
[contrib_T2_pca, contrib_SPE_pca] = compute_contributions('pca', first_alarm_pca, Z, scores_pca, latent, coeffs, hat_Z, [], []);

% k-PCA Contributions
[contrib_T2_kpca, contrib_SPE_kpca] = compute_contributions('kpca', first_alarm_kpca, Z, scores_kpca, lambda, [], [], kernel_info, alpha);

% Normalize to percentage
contrib_T2_pca_pct = 100 * contrib_T2_pca / sum(contrib_T2_pca);
contrib_SPE_pca_pct = 100 * contrib_SPE_pca / sum(contrib_SPE_pca);
contrib_T2_kpca_pct = 100 * contrib_T2_kpca / sum(contrib_T2_kpca);
contrib_SPE_kpca_pct = 100 * contrib_SPE_kpca / sum(contrib_SPE_kpca);

% Plot PCA: T² and SPE
figure('Position', [100, 100, 1400, 550]);
subplot(1,2,1);
bar(contrib_T2_pca_pct, 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'k', 'LineWidth', 0.5);
xticks(1:length(scale_info.kept_names));
xticklabels(scale_info.kept_names);
xtickangle(45);
ylabel('Contribution (%)', 'FontSize', 13, 'FontWeight', 'bold');
title('PCA: \bfT^2 Contributions at WT14 First Alarm', 'FontSize', 14, 'FontWeight', 'bold');
grid on; set(gca, 'FontSize', 12);
ylim([0 max(contrib_T2_pca_pct)*1.1]);

subplot(1,2,2);
bar(contrib_SPE_pca_pct, 'FaceColor', [0.8 0.4 0.2], 'EdgeColor', 'k', 'LineWidth', 0.5);
xticks(1:length(scale_info.kept_names));
xticklabels(scale_info.kept_names);
xtickangle(45);
ylabel('Contribution (%)', 'FontSize', 13, 'FontWeight', 'bold');
title('PCA: \bfSPE Contributions at WT14 First Alarm', 'FontSize', 14, 'FontWeight', 'bold');
grid on; set(gca, 'FontSize', 12);
ylim([0 max(contrib_SPE_pca_pct)*1.1]);
saveas(gcf, '../figures/pca_contributions_T2_SPE_WT14.png');

% Plot k-PCA: T² and SPE
figure('Position', [120, 120, 1400, 550]);
subplot(1,2,1);
bar(contrib_T2_kpca_pct, 'FaceColor', [0.3 0.7 0.3], 'EdgeColor', 'k', 'LineWidth', 0.5);
xticks(1:length(scale_info.kept_names));
xticklabels(scale_info.kept_names);
xtickangle(45);
ylabel('Contribution (%)', 'FontSize', 13, 'FontWeight', 'bold');
title('k-PCA: \bfT^2 Contributions at WT14 First Alarm', 'FontSize', 14, 'FontWeight', 'bold');
grid on; set(gca, 'FontSize', 12);
ylim([0 max(contrib_T2_kpca_pct)*1.1]);

subplot(1,2,2);
bar(contrib_SPE_kpca_pct, 'FaceColor', [0.9 0.6 0.2], 'EdgeColor', 'k', 'LineWidth', 0.5);
xticks(1:length(scale_info.kept_names));
xticklabels(scale_info.kept_names);
xtickangle(45);
ylabel('Contribution (%)', 'FontSize', 13, 'FontWeight', 'bold');
title('k-PCA: \bfSPE Contributions at WT14 First Alarm', 'FontSize', 14, 'FontWeight', 'bold');
grid on; set(gca, 'FontSize', 12);
ylim([0 max(contrib_SPE_kpca_pct)*1.1]);
saveas(gcf, '../figures/kpca_contributions_T2_SPE_WT14.png');

% Print top-3 contributors for report
[~, top_T2_pca] = sort(contrib_T2_pca_pct, 'descend');
[~, top_SPE_pca] = sort(contrib_SPE_pca_pct, 'descend');
[~, top_T2_kpca] = sort(contrib_T2_kpca_pct, 'descend');
[~, top_SPE_kpca] = sort(contrib_SPE_kpca_pct, 'descend');

fprintf('\n=== Top-3 Contributors (WT14 First Alarm) ===\n');
fprintf('PCA T²:  %s (%.1f%%), %s (%.1f%%), %s (%.1f%%)\n', ...
    scale_info.kept_names{top_T2_pca(1)}, contrib_T2_pca_pct(top_T2_pca(1)), ...
    scale_info.kept_names{top_T2_pca(2)}, contrib_T2_pca_pct(top_T2_pca(2)), ...
    scale_info.kept_names{top_T2_pca(3)}, contrib_T2_pca_pct(top_T2_pca(3)));

fprintf('PCA SPE: %s (%.1f%%), %s (%.1f%%), %s (%.1f%%)\n', ...
    scale_info.kept_names{top_SPE_pca(1)}, contrib_SPE_pca_pct(top_SPE_pca(1)), ...
    scale_info.kept_names{top_SPE_pca(2)}, contrib_SPE_pca_pct(top_SPE_pca(2)), ...
    scale_info.kept_names{top_SPE_pca(3)}, contrib_SPE_pca_pct(top_SPE_pca(3)));

fprintf('k-PCA T²:  %s (%.1f%%), %s (%.1f%%), %s (%.1f%%)\n', ...
    scale_info.kept_names{top_T2_kpca(1)}, contrib_T2_kpca_pct(top_T2_kpca(1)), ...
    scale_info.kept_names{top_T2_kpca(2)}, contrib_T2_kpca_pct(top_T2_kpca(2)), ...
    scale_info.kept_names{top_T2_kpca(3)}, contrib_T2_kpca_pct(top_T2_kpca(3)));

fprintf('k-PCA SPE: %s (%.1f%%), %s (%.1f%%), %s (%.1f%%)\n', ...
    scale_info.kept_names{top_SPE_kpca(1)}, contrib_SPE_kpca_pct(top_SPE_kpca(1)), ...
    scale_info.kept_names{top_SPE_kpca(2)}, contrib_SPE_kpca_pct(top_SPE_kpca(2)), ...
    scale_info.kept_names{top_SPE_kpca(3)}, contrib_SPE_kpca_pct(top_SPE_kpca(3)));

fprintf('\nMSPC analysis complete! Check ../figures/ for all plots.\n');

%% === EXTRA: WT39 Contribution Analysis (PCA & k-PCA) ===
fprintf('\n=== Diagnostics: WT39 Contributions (PCA & k-PCA) ===\n');

% First alarm (or representative) in WT39 for PCA
alarms_pca_f2 = (T2_pca(idx_f2) > T2_limit_pca) | (SPE_pca(idx_f2) > SPE_limit_pca);
if any(alarms_pca_f2)
    posRel_pca_f2 = find(alarms_pca_f2, 1, 'first');
else
    score_pca_f2 = (T2_pca(idx_f2)/max(T2_limit_pca,eps)) + (SPE_pca(idx_f2)/max(SPE_limit_pca,eps));
    [~, posRel_pca_f2] = max(score_pca_f2);
end
alarm_idx_pca_wt39 = idx_f2(posRel_pca_f2);

% First alarm (or representative) in WT39 for k-PCA
alarms_kpca_f2 = (T2_kpca(idx_f2) > T2_limit_kpca) | (SPE_kpca(idx_f2) > SPE_limit_kpca);
if any(alarms_kpca_f2)
    posRel_kpca_f2 = find(alarms_kpca_f2, 1, 'first');
else
    score_kpca_f2 = (T2_kpca(idx_f2)/max(T2_limit_kpca,eps)) + (SPE_kpca(idx_f2)/max(SPE_limit_kpca,eps));
    [~, posRel_kpca_f2] = max(score_kpca_f2);
end
alarm_idx_kpca_wt39 = idx_f2(posRel_kpca_f2);

% Compute contributions (PCA and k-PCA)
[contrib_T2_pca_wt39,  contrib_SPE_pca_wt39]  = compute_contributions( ...
    'pca',  alarm_idx_pca_wt39,  Z, scores_pca,  latent, coeffs, hat_Z, [], []);
[contrib_T2_kpca_wt39, contrib_SPE_kpca_wt39] = compute_contributions( ...
    'kpca', alarm_idx_kpca_wt39, Z, scores_kpca, lambda, [], [], kernel_info, alpha);

% Normalize to %
pct_T2_pca_wt39   = 100 * contrib_T2_pca_wt39   / (sum(contrib_T2_pca_wt39)   + eps);
pct_SPE_pca_wt39  = 100 * contrib_SPE_pca_wt39  / (sum(contrib_SPE_pca_wt39)  + eps);
pct_T2_kpca_wt39  = 100 * contrib_T2_kpca_wt39  / (sum(contrib_T2_kpca_wt39)  + eps);
pct_SPE_kpca_wt39 = 100 * contrib_SPE_kpca_wt39 / (sum(contrib_SPE_kpca_wt39) + eps);

% Labels for kPCA T^2 bars (components vs variables)
if length(pct_T2_kpca_wt39) == length(scale_info.kept_names)
    labels_kpca_T2_wt39 = scale_info.kept_names;
else
    labels_kpca_T2_wt39 = arrayfun(@(k) sprintf('PC%d', k), 1:length(pct_T2_kpca_wt39), 'UniformOutput', false);
end

% WT39 - PCA figure (T^2 and SPE)
figure('Position',[120,120,1400,550]);
subplot(1,2,1);
bar(pct_T2_pca_wt39, 'FaceColor',[0.2 0.6 0.8], 'EdgeColor','k'); grid on;
xticks(1:length(scale_info.kept_names)); xticklabels(scale_info.kept_names); xtickangle(45);
ylabel('Contribution (%)','FontSize',13,'FontWeight','bold');
title('PCA: T^2 Contributions at WT39 First Alarm','FontSize',14,'FontWeight','bold');

subplot(1,2,2);
bar(pct_SPE_pca_wt39, 'FaceColor',[0.8 0.4 0.2], 'EdgeColor','k'); grid on;
xticks(1:length(scale_info.kept_names)); xticklabels(scale_info.kept_names); xtickangle(45);
ylabel('Contribution (%)','FontSize',13,'FontWeight','bold');
title('PCA: SPE (Q) Contributions at WT39 First Alarm','FontSize',14,'FontWeight','bold');

saveas(gcf, '../figures/pca_contributions_T2_SPE_WT39.png');

% WT39 - kPCA figure (T^2 and SPE)
figure('Position',[140,140,1400,550]);
subplot(1,2,1);
bar(pct_T2_kpca_wt39, 'FaceColor',[0.3 0.7 0.3], 'EdgeColor','k'); grid on;
xticks(1:length(pct_T2_kpca_wt39)); xticklabels(labels_kpca_T2_wt39); xtickangle(45);
ylabel('Contribution (%)','FontSize',13,'FontWeight','bold');
title('k-PCA: T^2 Contributions at WT39 First Alarm','FontSize',14,'FontWeight','bold');

subplot(1,2,2);
bar(pct_SPE_kpca_wt39, 'FaceColor',[0.9 0.6 0.2], 'EdgeColor','k'); grid on;
xticks(1:length(scale_info.kept_names)); xticklabels(scale_info.kept_names); xtickangle(45);
ylabel('Contribution (%)','FontSize',13,'FontWeight','bold');
title('k-PCA: SPE (Q) Contributions at WT39 First Alarm','FontSize',14,'FontWeight','bold');

saveas(gcf, '../figures/kpca_contributions_T2_SPE_WT39.png');