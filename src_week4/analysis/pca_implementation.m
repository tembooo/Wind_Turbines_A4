%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Arman Golbidi
% This function performs PCA with healthy-based scaling.
% - Drop zero-variance-in-healthy columns
% - Scale all rows with healthy μ,σ
% - Fit PCA on Healthy block only, project all rows
% - Mathematical description: Z = (X - μ_h) / σ_h, PCA: X = TP' + E, where T = scores, P = loadings

function [coeffs, scores, latent, explained, scale_info] = pca_implementation(data_matrix, n_healthy, var_names)
% PCA with healthy-based scaling

    fprintf('running PCA on %d observations and %d variables...\n', size(data_matrix, 1), size(data_matrix, 2));

    %% Data quality check
    nan_count = sum(isnan(data_matrix(:)));
    inf_count = sum(isinf(data_matrix(:)));
    fprintf('checking data quality...\n  missing values: %d\n  infinite values: %d\n', nan_count, inf_count);
    if nan_count > 0 || inf_count > 0
        error('Data contains missing or infinite values.');
    end

    %% Healthy-based scaling
    n_vars_in = size(data_matrix, 2);

    if nargin < 3 || isempty(var_names)
        var_names = arrayfun(@(k) sprintf('v%d', k), 1:n_vars_in, 'UniformOutput', false);
    end

    mu_h_full = mean(data_matrix(1:n_healthy, :), 1);
    sd_h_full = std(data_matrix(1:n_healthy, :), 0, 1);

    valid_cols = sd_h_full > 0;
    dropped_idx = find(~valid_cols);
    kept_idx = find(valid_cols);

    fprintf('Debug: n_vars_in=%d, length(mu_h_full)=%d, length(sd_h_full)=%d\n', n_vars_in, length(mu_h_full), length(sd_h_full));
    fprintf('Debug: length(kept_idx)=%d, length(dropped_idx)=%d\n', length(kept_idx), length(dropped_idx));
    fprintf('Debug: length(var_names)=%d, length(var_names(kept_idx))=%d, length(var_names(dropped_idx))=%d\n', ...
            length(var_names), length(var_names(kept_idx)), length(var_names(dropped_idx)));

    data_matrixK = data_matrix(:, kept_idx);
    mu_h = mu_h_full(kept_idx);  % Means for kept variables only
    sd_h = sd_h_full(kept_idx);  % SDs for kept variables only

    Z = (data_matrixK - mu_h) ./ sd_h;

    % Debug lengths before struct
    fprintf('Debug: length(mu_h)=%d, length(sd_h)=%d, length(kept_idx)=%d, length(dropped_idx)=%d\n', ...
            length(mu_h), length(sd_h), length(kept_idx), length(dropped_idx));
    fprintf('Debug: length(kept_names)=%d, length(dropped_names)=%d\n', ...
            length(var_names(kept_idx)), length(var_names(dropped_idx)));

    % Create structure incrementally to avoid dimension mismatch
    scale_info = struct();
    scale_info.n_vars_in = n_vars_in;
    scale_info.mu_h_full = mu_h_full;
    scale_info.sd_h_full = sd_h_full;
    scale_info.mu_h = mu_h;
    scale_info.sd_h = sd_h;
    scale_info.kept_idx = kept_idx;
    scale_info.dropped_idx = dropped_idx;
    scale_info.kept_names = var_names(kept_idx);
    scale_info.dropped_names = var_names(dropped_idx);

    fprintf('scaling (healthy-based): kept %d of %d columns; dropped %d (sd_h==0)\n', ...
            numel(kept_idx), n_vars_in, numel(dropped_idx));

    fprintf('preprocessed data check:\n  size (all rows, kept vars): %d x %d\n  rank(all): %d\n  condition number(all): %.2e\n', ...
            size(Z, 1), size(Z, 2), rank(Z), cond(Z));

    %% PCA on healthy block only
    [coeffs, scores, latent, ~, explained] = pca(Z(1:n_healthy, :));

    fprintf('running pca on HEALTHY block only...\n');
    fprintf('pca(healthy) completed successfully\n');
    fprintf('  coeffs size: %d x %d\n  healthy scores size: %d x %d\n  latent length: %d\n', ...
            size(coeffs), size(scores, 1), size(scores, 2), length(latent));

    %% Project all data
    scores = Z * coeffs;

    fprintf('[check] mean PC1/PC2 on Healthy ≈ [%.3f, %.3f]\n', mean(scores(1:n_healthy, 1)), mean(scores(1:n_healthy, 2)));

    %% Summary
    total_var = sum(latent);
    kaiser = latent > 1;
    fprintf('pca summary (healthy-based model):\n  total components: %d\n  total variance (sum of eigenvalues): %.2f\n  kaiser rule suggests keeping %d components\n', ...
            length(latent), total_var, sum(kaiser));

    [~, idx] = sort(abs(coeffs), 'descend');
    for i = 1:3
        fprintf('  pc%d (%.1f%% variance): ', i, explained(i));
        top_vars = idx(1:3, i);
        for j = 1:3
            fprintf('%s(%.2f) ', scale_info.kept_names{top_vars(j)}, coeffs(top_vars(j), i));
        end
        fprintf('\n');
    end
end
