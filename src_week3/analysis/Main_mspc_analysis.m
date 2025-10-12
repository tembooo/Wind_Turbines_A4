%% MSPC analysis for wind turbines - ADML course project (PCA vs k-PCA)

clear; clc; close all;
addpath('../analysis');       % analysis module
addpath('../data_loading');   % data loading module

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

%% Diagnostics: Contributions (example for first alarm in WT14)
fprintf('\n=== Diagnostics: Contributions ===\n');

first_alarm_pca = n_healthy + ttd_pca_f1;
% Assuming this is around line 100
[contrib_T2_pca, contrib_SPE_pca] = compute_contributions('pca', first_alarm_pca, Z, scores_pca, latent, coeffs, [], []);

first_alarm_kpca = n_healthy + ttd_kpca_f1;
[contrib_T2_kpca, contrib_SPE_kpca] = compute_contributions('kpca', first_alarm_kpca, Z, scores_kpca, lambda, [], [], kernel_info, alpha);

% Sort and print top 3 for report
[~, top_SPE_pca] = sort(contrib_SPE_pca, 'descend');
fprintf('PCA Top SPE Contributors (WT14 first alarm): %s(%.1f%%), %s(%.1f%%), %s(%.1f%%)\n', ...
        scale_info.kept_names{top_SPE_pca(1)}, contrib_SPE_pca(top_SPE_pca(1))/sum(contrib_SPE_pca)*100, ...
        scale_info.kept_names{top_SPE_pca(2)}, contrib_SPE_pca(top_SPE_pca(2))/sum(contrib_SPE_pca)*100, ...
        scale_info.kept_names{top_SPE_pca(3)}, contrib_SPE_pca(top_SPE_pca(3))/sum(contrib_SPE_pca)*100);

% Similar for k-PCA and T2

%% Control Charts Visualization
fprintf('\n=== Generating Control Charts ===\n');

% PCA for WT14
figure('Position',[100,100,800,600]);
subplot(2,1,1); plot(T2_pca(idx_f1), 'b-'); hold on;
plot([1 n_faulty1], [T2_limit_pca T2_limit_pca], 'r--');
title('PCA T^2 Control Chart - WT14'); xlabel('Sample'); ylabel('T^2'); grid on;

subplot(2,1,2); plot(SPE_pca(idx_f1), 'b-'); hold on;
plot([1 n_faulty1], [SPE_limit_pca SPE_limit_pca], 'r--');
title('PCA SPE Control Chart - WT14'); xlabel('Sample'); ylabel('SPE'); grid on;

% Similar figures for k-PCA WT14, PCA/k-PCA WT39

% Contribution bars (example)
figure('Position',[200,200,800,400]);
bar(contrib_SPE_pca); xticklabels(scale_info.kept_names); xtickangle(45);
title('PCA SPE Contributions - WT14 First Alarm'); xlabel('Sensor'); ylabel('Contribution');

% Repeat for others

fprintf('MSPC analysis complete! Check console/metrics and figures for report.\n');


































%% ===== Save all required figures (PCA vs k-PCA) =====
% Output folder for figures
out_dir = fullfile(pwd, 'figs');
if ~exist(out_dir, 'dir'); mkdir(out_dir); end

% ---------- Fig 1: PCA WT14 (T^2 & SPE) ----------
try
    exportgraphics(gcf, fullfile(out_dir,'Fig1_PCA_T2_SPE_WT14.png'), 'Resolution',300);
catch
    saveas(gcf, fullfile(out_dir,'Fig1_PCA_T2_SPE_WT14.png'));
end

% ---------- Fig 2: k-PCA WT14 (T^2 & SPE) ----------
figure('Position',[100,100,800,600]);
subplot(2,1,1); plot(T2_kpca(idx_f1), 'b-'); hold on;
plot([1 n_faulty1], [T2_limit_kpca T2_limit_kpca], 'r--');
title('k-PCA T^2 Control Chart - WT14'); xlabel('Sample'); ylabel('T^2'); grid on;

subplot(2,1,2); plot(SPE_kpca(idx_f1), 'b-'); hold on;
plot([1 n_faulty1], [SPE_limit_kpca SPE_limit_kpca], 'r--');
title('k-PCA SPE Control Chart - WT14'); xlabel('Sample'); ylabel('SPE'); grid on;

try
    exportgraphics(gcf, fullfile(out_dir,'Fig2_kPCA_T2_SPE_WT14.png'), 'Resolution',300);
catch
    saveas(gcf, fullfile(out_dir,'Fig2_kPCA_T2_SPE_WT14.png'));
end

% ---------- Fig 3: PCA WT39 (T^2 & SPE) ----------
figure('Position',[100,100,800,600]);
subplot(2,1,1); plot(T2_pca(idx_f2), 'b-'); hold on;
plot([1 n_faulty2], [T2_limit_pca T2_limit_pca], 'r--');
title('PCA T^2 Control Chart - WT39'); xlabel('Sample'); ylabel('T^2'); grid on;

subplot(2,1,2); plot(SPE_pca(idx_f2), 'b-'); hold on;
plot([1 n_faulty2], [SPE_limit_pca SPE_limit_pca], 'r--');
title('PCA SPE Control Chart - WT39'); xlabel('Sample'); ylabel('SPE'); grid on;

try
    exportgraphics(gcf, fullfile(out_dir,'Fig3_PCA_T2_SPE_WT39.png'), 'Resolution',300);
catch
    saveas(gcf, fullfile(out_dir,'Fig3_PCA_T2_SPE_WT39.png'));
end

% ---------- Fig 4: k-PCA WT39 (T^2 & SPE) ----------
figure('Position',[100,100,800,600]);
subplot(2,1,1); plot(T2_kpca(idx_f2), 'b-'); hold on;
plot([1 n_faulty2], [T2_limit_kpca T2_limit_kpca], 'r--');
title('k-PCA T^2 Control Chart - WT39'); xlabel('Sample'); ylabel('T^2'); grid on;

subplot(2,1,2); plot(SPE_kpca(idx_f2), 'b-'); hold on;
plot([1 n_faulty2], [SPE_limit_kpca SPE_limit_kpca], 'r--');
title('k-PCA SPE Control Chart - WT39'); xlabel('Sample'); ylabel('SPE'); grid on;

try
    exportgraphics(gcf, fullfile(out_dir,'Fig4_kPCA_T2_SPE_WT39.png'), 'Resolution',300);
catch
    saveas(gcf, fullfile(out_dir,'Fig4_kPCA_T2_SPE_WT39.png'));
end

% ---------- Fig 5: PCA Contributions (WT14, first alarm) ----------
try
    exportgraphics(gcf, fullfile(out_dir,'Fig5_PCA_SPE_Contributions_WT14.png'), 'Resolution',300);
catch
    saveas(gcf, fullfile(out_dir,'Fig5_PCA_SPE_Contributions_WT14.png'));
end
