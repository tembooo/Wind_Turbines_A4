%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Fasie Haider
% Performs contiguous 5-fold CV for FAR and ARL calculation on healthy data.
% Mathematical description: Contiguous folds preserve time (e.g., fold1: 1:fold_size, etc.).
%   FAR = mean(alarms on validation), ARL = average run length between alarms.
%   Alarms: T2 > limit or SPE > limit.

function [avg_FAR_T2, avg_FAR_SPE, avg_ARL] = validate_model(model_type, all_data, n_healthy, var_names, alpha_level, sigma)
% Contiguous 5-fold CV on healthy; returns avg FAR (T2/SPE) and ARL

    if nargin < 6, sigma = []; end  % For PCA, sigma not needed

    k = 5;
    nh = n_healthy;
    fold_size = floor(nh / k);
    FAR_T2 = zeros(k,1);
    FAR_SPE = zeros(k,1);
    ARL = zeros(k,1);

    for f = 1:k
        test_start = (f-1)*fold_size + 1;
        test_end = min(test_start + fold_size - 1, nh);
        test_idx = test_start:test_end;
        train_idx = setdiff(1:nh, test_idx);
        nt = length(train_idx);

        data_train = all_data(train_idx,:);
        data_test = all_data(test_idx,:);

        if strcmp(model_type, 'pca')
            [coeffs, ~, latent, ~, scale_info] = pca_implementation(data_train, nt, var_names); % Use full var_names
            a = 4;  % Optimize: tune via cumvar >=90% or Kaiser >1
            coeffs = coeffs(:,1:a);
            latent = latent(1:a);

            Z_train = (data_train(:, scale_info.kept_idx) - scale_info.mu_h) ./ scale_info.sd_h;
            scores_train = Z_train * coeffs;
            hat_Z_train = scores_train * coeffs';

            [T2_train, SPE_train, T2_limit, SPE_limit] = compute_stats('pca', scores_train, latent, alpha_level, nt, Z_train, hat_Z_train);

            Z_test = (data_test(:, scale_info.kept_idx) - scale_info.mu_h) ./ scale_info.sd_h;
            scores_test = Z_test * coeffs;
            hat_Z_test = scores_test * coeffs';
            [T2_test, SPE_test, ~, ~] = compute_stats('pca', scores_test, latent, alpha_level, [], Z_test, hat_Z_test);

        else  % kpca
            [alpha, lambda, ~, ~, scale_info, kernel_info] = kpca_implementation(data_train, nt, var_names, sigma); % Use full var_names
            a = 4;  % Same as PCA for fair comparison
            alpha = alpha(:,1:a);
            lambda = lambda(1:a);

            Z_train = (data_train(:, scale_info.kept_idx) - scale_info.mu_h) ./ scale_info.sd_h;
            Z_test = (data_test(:, scale_info.kept_idx) - scale_info.mu_h) ./ scale_info.sd_h;

            % Update kernel_info for training data
            kernel_info.Zh = Z_train;
            kernel_info.K = rbf_kernel(Z_train, Z_train, sigma);
            kernel_info.one_n = ones(nt, nt) / nt;
            kernel_info.mean_train = mean(kernel_info.K(:));

            % Compute scores for training and test sets
            scores_train = compute_scores_kpca(Z_train, kernel_info, alpha);
            scores_test = compute_scores_kpca(Z_test, kernel_info, alpha);

            % Compute stats for training to get limits
            [T2_train, SPE_train, T2_limit, SPE_limit] = compute_stats('kpca', scores_train, lambda, alpha_level, nt, Z_train, [], kernel_info);

            % Compute stats for test set
            [T2_test, SPE_test, ~, ~] = compute_stats('kpca', scores_test, lambda, alpha_level, length(test_idx), Z_test, [], kernel_info);
        end

        % Calculate alarms and performance metrics
        alarms = (T2_test > T2_limit) | (SPE_test > SPE_limit);
        FAR_T2(f) = mean(T2_test > T2_limit);
        FAR_SPE(f) = mean(SPE_test > SPE_limit);

        % === Compute Average Run Length (ARL) ===
        if any(alarms)
            pos  = find(alarms);                            % indices of alarms
            runs = diff([0; pos; length(alarms)+1]) - 1;    % number of non-alarm samples between alarms
            ARL(f) = mean(runs);                            % average run length
        else
            ARL(f) = length(alarms);                        % if no alarms, ARL = length of test block
        end

    end

    avg_FAR_T2 = mean(FAR_T2);
    avg_FAR_SPE = mean(FAR_SPE);
    avg_ARL = mean(ARL);
end

% Helper function to compute kPCA scores
function scores = compute_scores_kpca(Z, kernel_info, alpha)
    sigma = kernel_info.sigma;
    Zh = kernel_info.Zh;
    K_all = rbf_kernel(Z, Zh, sigma);
    one_n_t = ones(size(Z,1), size(Zh,1)) / size(Zh,1);
    tilde_K_all = K_all - one_n_t * kernel_info.K - K_all * kernel_info.one_n + one_n_t * kernel_info.K * kernel_info.one_n;
    scores = tilde_K_all * alpha;
end