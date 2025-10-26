%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Arman Golbidi
% This function performs kernel PCA with Gaussian (RBF) kernel, healthy-based scaling.
% - Drop zero-variance-in-healthy columns
% - Scale all rows with healthy μ,σ
% - Fit KPCA on Healthy block only, project all rows
% - Mathematical description: Gaussian kernel K(x_i,x_j) = exp(-||x_i - x_j||^2 / (2σ^2))
%   Centered kernel: \tilde{K} = K - 1_n K - K 1_n + 1_n K 1_n (double-centering)
%   Eigendecomposition: \tilde{K} = V Λ V^T, alpha = V * diag(1/sqrt(λ))
%   Scores for new data: t = \tilde{K}_{test} alpha

function [alpha, lambda, scores, var_explained, scale_info, kernel_info] = kpca_implementation(data_matrix, n_healthy, var_names, sigma)
% KPCA with Gaussian (RBF) kernel, healthy-based scaling
% sigma: kernel width for Gaussian kernel K(x,y) = exp(-||x-y||^2 / (2*sigma^2))

    fprintf('running KPCA (Gaussian RBF, sigma=%.2f) on %d observations and %d variables...\n', sigma, size(data_matrix, 1), size(data_matrix, 2));

    %% Data quality check: no NaNs allowed post-pretreatment
    nan_count = sum(isnan(data_matrix(:)));
    if nan_count > 0
        error('Data contains missing values after pretreatment. Run time_aware_preprocess first.');
    end

    %% Healthy-based scaling
    n_vars_in = size(data_matrix,2);

    if nargin < 3 || isempty(var_names)
        var_names = arrayfun(@(k) sprintf('v%d',k), 1:n_vars_in, 'UniformOutput', false);
    end

    mu_h_full = mean(data_matrix(1:n_healthy,:), 1);
    sd_h_full = std(data_matrix(1:n_healthy,:), 0, 1);

    valid_cols = sd_h_full > 0;
    dropped_idx = find(~valid_cols);
    kept_idx = find(valid_cols);

    data_matrixK = data_matrix(:, kept_idx);
    mu_h = mu_h_full(kept_idx);
    sd_h = sd_h_full(kept_idx);

    Z = (data_matrixK - mu_h) ./ sd_h;  % All rows scaled by healthy stats

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

    %% KPCA on healthy block only
    nh = n_healthy;
    Zh = Z(1:nh, :);

    K = rbf_kernel(Zh, Zh, sigma);  % Gaussian (RBF) kernel matrix

    one_n = ones(nh, nh) / nh;
    tilde_K = K - one_n * K - K * one_n + one_n * K * one_n;  % Centered kernel (double-centering)

    [V, L] = eig(tilde_K);
    [l, idx] = sort(diag(L), 'descend');
    V = V(:, idx);

    l = l(l > 1e-10);  % Discard tiny eigenvalues
    a = length(l);
    V = V(:,1:a);

    alpha = V * diag(1 ./ sqrt(l));  % Normalized eigenvectors

    lambda = l / nh;  % Variances (eigenvalues scaled)

    var_explained = 100 * lambda / sum(lambda);

    %% Project all data
    n = size(Z,1);
    K_all = rbf_kernel(Z, Zh, sigma);

    one_n_t = ones(n, nh) / nh;
    tilde_K_all = K_all - one_n_t * K - K_all * one_n + one_n_t * K * one_n;

    scores = tilde_K_all * alpha;

    %% Kernel info for later SPE/T2 contributions
    mean_train = mean(K(:));  % For SPE approximation
    kernel_info = struct('sigma', sigma, 'mean_train', mean_train, 'Zh', Zh, 'one_n', one_n, 'K', K);

    fprintf('kPCA completed: %d components retained, first 3 explain [%.1f%%, %.1f%%, %.1f%%]\n', ...
            a, var_explained(1), var_explained(min(2,a)), var_explained(min(3,a)));

end

function K = rbf_kernel(X1, X2, sigma)
% Gaussian (RBF) kernel: K(x1, x2) = exp(-||x1 - x2||^2 / (2 * sigma^2))
    n1 = size(X1,1);
    n2 = size(X2,1);
    norm1 = sum(X1.^2,2);
    norm2 = sum(X2.^2,2);
    dist = repmat(norm1,1,n2) + repmat(norm2',n1,1) - 2 * X1 * X2';
    K = exp( - dist / (2 * sigma^2 ) );
end
