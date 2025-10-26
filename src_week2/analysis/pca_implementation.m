%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Fasie Haider
% Performs PCA with healthy-based scaling.
% - Drop zero-variance-in-healthy columns
% - Scale all rows with healthy μ,σ
% - Fit PCA on Healthy block only, project all rows
% - Returns coeffs, scores, explained variance, scale_info


function [coeffs, scores, latent, explained, scale_info] = pca_implementation(data_matrix, n_healthy, var_names)
% Preprocess via healthy-based standardization and run MATLAB's PCA.
% - Compute mean/std on the healthy block only.
% - Drop variables with zero variance in healthy.
% - Standardize ALL rows using healthy μ, σ.
% - Fit PCA on the Healthy block ONLY; then project ALL rows to that model.
% - No time-step removal here; pretreatment must be done upstream.

    fprintf('running PCA on %d observations and %d variables...\n', ...
            size(data_matrix, 1), size(data_matrix, 2));

    %% Data quality check: do NOT drop time steps here
    fprintf('checking data quality...\n');
    nan_count = sum(isnan(data_matrix(:)));
    inf_count = sum(isinf(data_matrix(:)));
    fprintf('  missing values: %d\n', nan_count);
    fprintf('  infinite values: %d\n', inf_count);
    if nan_count > 0
        error(['Data contains %d missing values after pretreatment. ', ...
               'Run time_aware_preprocess first; PCA will not drop rows.'], nan_count);
    end

    %% Healthy-based scaling (per instructor guidance)
    n_vars_in = size(data_matrix,2);

    % Ensure var_names is available for reporting
    if nargin < 3 || isempty(var_names)
        var_names = arrayfun(@(k) sprintf('v%d',k), 1:n_vars_in, 'UniformOutput', false);
    else
        var_names = var_names(:).';
        if numel(var_names) ~= n_vars_in
            error('Length of var_names (%d) does not match number of columns (%d).', ...
                  numel(var_names), n_vars_in);
        end
    end

    % 1) Healthy μ and σ, computed on the first n_healthy rows
    mu_h_full = mean(data_matrix(1:n_healthy,:), 1);
    sd_h_full = std( data_matrix(1:n_healthy,:), 0, 1);

    % 2) Drop zero-variance-in-healthy variables
    valid_cols  = sd_h_full > 0;
    dropped_idx = find(~valid_cols);
    kept_idx    = find( valid_cols);

    data_matrixK = data_matrix(:, kept_idx);
    mu_h         = mu_h_full(kept_idx);
    sd_h         = sd_h_full(kept_idx);

    % 3) Standardize ALL rows with healthy μ,σ
    Z = (data_matrixK - mu_h) ./ sd_h;   % healthy-based autoscaling

    % Pack scaling info for reporting and plotting
    scale_info = struct();
    scale_info.n_vars_in      = n_vars_in;
    scale_info.mu_h_full      = mu_h_full;
    scale_info.sd_h_full      = sd_h_full;
    scale_info.kept_idx       = kept_idx;
    scale_info.dropped_idx    = dropped_idx;
    scale_info.kept_names     = var_names(kept_idx);
    scale_info.dropped_names  = var_names(dropped_idx);

    fprintf('scaling (healthy-based): kept %d of %d columns; dropped %d (sd_h==0)\n', ...
            numel(kept_idx), n_vars_in, numel(dropped_idx));

    %% Final sanity check on scaled matrix
    fprintf('preprocessed data check:\n');
    fprintf('  size (all rows, kept vars): %d x %d\n', size(Z));
    if any(isnan(Z(:))) || any(isinf(Z(:)))
        fprintf('  warning: preprocessed data contains NaN/Inf values! Replacing with 0.\n');
        Z(isnan(Z)) = 0;
        Z(isinf(Z)) = 0;
    end
    try
        fprintf('  rank(all): %d\n', rank(Z));
        fprintf('  condition number(all): %.2e\n', cond(Z));
    catch
        fprintf('  warning: could not compute rank/cond on all rows.\n');
    end

    %% === Fit PCA on HEALTHY ONLY, then project ALL rows ===
    Zh = Z(1:n_healthy, :);  % Healthy block (already healthy-scaled)

    fprintf('running pca on HEALTHY block only...\n');
    try
        % coefficients, healthy scores, eigenvalues, explained %, and mean used by PCA
        [coeffs, scores_h, latent, ~, explained, mu_center] = pca(Zh);
        fprintf('pca(healthy) completed successfully\n');
        fprintf('  coeffs size: %d x %d\n', size(coeffs));
        fprintf('  healthy scores size: %d x %d\n', size(scores_h));
        fprintf('  latent length: %d\n', length(latent));
    catch ME
        error('pca(healthy) failed: %s', ME.message);
    end

    % Project ALL rows using the HEALTHY model (same coeffs & centering)
    % scores = (Z - mu_center) * coeffs;
    scores = bsxfun(@minus, Z, mu_center) * coeffs;

    % Quick sanity: Healthy means in PC1/PC2 should be ~0
    mH = mean(scores(1:n_healthy, 1:min(2,size(scores,2))), 1);
    fprintf('[check] mean PC1/PC2 on Healthy ≈ [%.3f, %.3f]\n', mH(1), mH(min(2,end)));

    %% Summary + quick top contributors log using kept variable names
    n_components = length(latent);
    fprintf('pca summary (healthy-based model):\n');
    fprintf('  total components: %d\n', n_components);
    fprintf('  total variance (sum of eigenvalues): %.2f\n', sum(latent));

    % Kaiser rule: eigenvalue > 1 (since columns were z-scored by healthy stats)
    kaiser_components = sum(latent > 1);
    fprintf('  kaiser rule suggests keeping %d components\n', kaiser_components);

    % Top contributing variables (first up to 3 PCs)
    fprintf('top contributing variables for first 3 components (by |loading|):\n');
    for pc = 1:min(3, n_components)
        [~, top_vars] = sort(abs(coeffs(:, pc)), 'descend');
        fprintf('  pc%d (%.1f%% variance):', pc, explained(pc));
        for i = 1:min(3, numel(top_vars))
            var_idx = top_vars(i);
            fprintf(' %s(%.2f)', scale_info.kept_names{var_idx}, coeffs(var_idx, pc));
        end
        fprintf('\n');
    end
end




%% my old version


%{
function [coeffs, scores, latent, explained, scale_info] = pca_implementation(data_matrix, n_healthy, var_names)
% Preprocess via healthy-based standardization and run MATLAB's PCA.
% - Compute mean/std on the healthy block only.
% - Drop variables with zero variance in healthy.
% - Standardize ALL rows using healthy μ, σ.
% - No time-step removal here; pretreatment must be done upstream.

    fprintf('running PCA on %d observations and %d variables...\n', ...
            size(data_matrix, 1), size(data_matrix, 2));

    %% Data quality check: do NOT drop time steps here
    fprintf('checking data quality...\n');
    nan_count = sum(isnan(data_matrix(:)));
    inf_count = sum(isinf(data_matrix(:)));
    fprintf('  missing values: %d\n', nan_count);
    fprintf('  infinite values: %d\n', inf_count);
    if nan_count > 0
        error(['Data contains %d missing values after pretreatment. ', ...
               'Run time_aware_preprocess first; PCA will not drop rows.'], nan_count);
    end

    %% Healthy-based scaling (per instructor guidance)
    n_vars_in  = size(data_matrix,2);

    % Ensure var_names is available for reporting
    if nargin < 3 || isempty(var_names)
        var_names = arrayfun(@(k) sprintf('v%d',k), 1:n_vars_in, 'UniformOutput', false);
    else
        var_names = var_names(:).';
        if numel(var_names) ~= n_vars_in
            error('Length of var_names (%d) does not match number of columns (%d).', numel(var_names), n_vars_in);
        end
    end

    % 1) Healthy μ and σ, computed on the first n_healthy rows
    mu_h_full = mean(data_matrix(1:n_healthy,:), 1);
    sd_h_full = std( data_matrix(1:n_healthy,:), 0, 1);

    % 2) Drop zero-variance-in-healthy variables
    valid_cols   = sd_h_full > 0;
    dropped_idx  = find(~valid_cols);
    kept_idx     = find( valid_cols);

    data_matrixK = data_matrix(:, kept_idx);
    mu_h         = mu_h_full(kept_idx);
    sd_h         = sd_h_full(kept_idx);

    % 3) Standardize ALL rows with healthy μ,σ
    data_preprocessed = (data_matrixK - mu_h) ./ sd_h;

    % Pack scaling info for reporting and plotting
    scale_info = struct();
    scale_info.n_vars_in      = n_vars_in;
    scale_info.mu_h_full      = mu_h_full;
    scale_info.sd_h_full      = sd_h_full;
    scale_info.kept_idx       = kept_idx;
    scale_info.dropped_idx    = dropped_idx;
    scale_info.kept_names     = var_names(kept_idx);
    scale_info.dropped_names  = var_names(dropped_idx);

    fprintf('scaling (healthy-based): kept %d of %d columns; dropped %d (sd_h==0)\n', ...
        numel(kept_idx), n_vars_in, numel(dropped_idx));

    %% Final sanity check
    fprintf('preprocessed data check:\n');
    fprintf('  size: %d x %d\n', size(data_preprocessed));
    if any(isnan(data_preprocessed(:))) || any(isinf(data_preprocessed(:)))
        fprintf('  warning: preprocessed data contains NaN/Inf values!\n');
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

    %% Run PCA
    fprintf('running pca...\n');
    try
        [coeffs, scores, latent] = pca(data_preprocessed);
        fprintf('pca function completed successfully\n');
        fprintf('  coeffs size: %d x %d\n', size(coeffs));
        fprintf('  scores size: %d x %d\n', size(scores));
        fprintf('  latent length: %d\n', length(latent));

        % Explained variance in percent
        explained = 100 * latent / sum(latent);
        fprintf('  calculated explained variance\n');
    catch ME
        fprintf('error in pca function: %s\n', ME.message);
        error('pca failed: %s', ME.message);
    end

    %% Summary + quick top contributors log using kept variable names
    n_components = length(latent);
    fprintf('pca completed successfully!\n');
    fprintf('  total components: %d\n', n_components);
    fprintf('  total variance (sum of eigenvalues): %.2f\n', sum(latent));

    % Kaiser rule: eigenvalue > 1
    kaiser_components = sum(latent > 1);
    fprintf('  kaiser rule suggests keeping %d components\n', kaiser_components);

    % Top contributing variables (first up to 3 PCs)
    fprintf('top contributing variables for first 3 components (by |loading|):\n');
    for pc = 1:min(3, n_components)
        [~, top_vars] = sort(abs(coeffs(:, pc)), 'descend');
        fprintf('  pc%d (%.1f%% variance):', pc, explained(pc));
        for i = 1:min(3, numel(top_vars))
            var_idx = top_vars(i);
            fprintf(' %s(%.2f)', scale_info.kept_names{var_idx}, coeffs(var_idx, pc));
        end
        fprintf('\n');
    end

end
%}