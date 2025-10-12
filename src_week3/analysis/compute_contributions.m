%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Fasie Haider — Debug Print Version
% Computes variable contributions to T2 and SPE for PCA and kPCA models.
% Includes diagnostic printouts to confirm matrix dimensions and alignment.

function [contrib_T2, contrib_SPE] = compute_contributions(model_type, alarm_idx, Z, scores, latent, coeffs, hat_Z, kernel_info, alpha)

    %=== Check alarm index validity ===%
    if isempty(alarm_idx) || alarm_idx > size(Z, 1)
        error('Invalid alarm index');
    end

    n_vars = size(Z, 2);
    contrib_T2 = zeros(1, n_vars);
    contrib_SPE = zeros(1, n_vars);

    eps0 = eps; % small constant for numerical stability

    %=== Print debug info header ===%
    fprintf('\n[DEBUG] Model Type: %s\n', model_type);
    fprintf('[DEBUG] Alarm index: %d\n', alarm_idx);
    fprintf('[DEBUG] Z size: %dx%d\n', size(Z,1), size(Z,2));
    fprintf('[DEBUG] Scores size: %dx%d\n', size(scores,1), size(scores,2));
    fprintf('[DEBUG] Latent size: %dx%d\n', size(latent,1), size(latent,2));

    if strcmp(model_type, 'pca')
        fprintf('[DEBUG] --- PCA branch ---\n');
        fprintf('[DEBUG] Coeffs size: %dx%d\n', size(coeffs,1), size(coeffs,2));

        %===== PCA: T2 Contribution =====%
        t = scores(alarm_idx, :);
        Lambda_inv = diag(1 ./ latent);
        T2_base = t * Lambda_inv * t';  % Scalar
        P = coeffs;

        % Weighted contributions per variable
        weighted_t = (t.^2) ./ latent';
        contrib_T2 = sum((P.^2) .* weighted_t, 2)';
        contrib_T2 = abs(contrib_T2);
        contrib_T2 = contrib_T2 / (sum(contrib_T2) + eps0) * T2_base;

        %===== PCA: SPE Contribution =====%
        if isempty(hat_Z)
            hat_Z = scores * coeffs';
        end
        residual = Z(alarm_idx, :) - hat_Z(alarm_idx, :);
        contrib_SPE = residual .^ 2;
        SPE_base = sum(contrib_SPE);

        contrib_SPE = abs(contrib_SPE) / (sum(contrib_SPE) + eps0) * SPE_base;

    else
        fprintf('[DEBUG] --- kPCA branch ---\n');
        if isempty(kernel_info)
            error('kernel_info required for kPCA contributions');
        end
        if isempty(alpha)
            error('alpha required for kPCA contributions');
        end

        %=== Debug kernel_info ===%
        fields = fieldnames(kernel_info);
        fprintf('[DEBUG] kernel_info fields: %s\n', strjoin(fields, ', '));
        if isfield(kernel_info, 'Zh')
            fprintf('[DEBUG] Zh size: %dx%d\n', size(kernel_info.Zh,1), size(kernel_info.Zh,2));
        end
        if isfield(kernel_info, 'K')
            fprintf('[DEBUG] K size: %dx%d\n', size(kernel_info.K,1), size(kernel_info.K,2));
        end

        %=== Shapes of alpha ===%
        fprintf('[DEBUG] Alpha size: %dx%d\n', size(alpha,1), size(alpha,2));

        sigma = kernel_info.sigma;
        Zh = kernel_info.Zh;
        m = size(Zh, 1);

        %=== Auto-fix alpha shape ===%
        [ra, ca] = size(alpha);
        if ra == m
            alpha_mat = alpha;
        elseif ca == m
            alpha_mat = alpha';
        else
            error('[DEBUG] alpha shape incompatible with Zh (%d vs %d)', m, ra);
        end

        %=== Centering preparation ===%
        one_n_vec = ones(m,1)/m;
        one_n_mat = ones(m)/m;
        one_n_t = one_n_vec';

        if isfield(kernel_info, 'K')
            K_train = kernel_info.K;
        else
            error('kernel_info.K missing.');
        end

        %=== Compute kernel row for alarm sample ===%
        x_alarm = Z(alarm_idx, :);
        K_all_row = rbf_kernel(x_alarm, Zh, sigma);  % 1 x m
        fprintf('[DEBUG] K_all_row size: %dx%d\n', size(K_all_row,1), size(K_all_row,2));

        %=== Center the kernel ===%
        tilde_K = K_all_row ...
            - (one_n_t * K_train) ...
            - (K_all_row * one_n_mat) ...
            + (one_n_t * K_train * one_n_mat);

        %=== Projection of alarm into kernel space ===%
        t = tilde_K * alpha_mat;  % 1 x A

        %=== SPE base computation ===%
        k_xx = rbf_kernel(x_alarm, x_alarm, sigma);  % scalar
        k_tilde_ii = k_xx - 2*mean(K_all_row) + mean(K_train(:));
        SPE_base = k_tilde_ii - (t * t');  % scalar
        fprintf('[DEBUG] SPE_base value: %.6f\n', SPE_base);

        %=== Gradient approximation for variable contribution ===%
        grad = zeros(1, n_vars);
        for j = 1:n_vars
            dK = zeros(1, m);
            for i = 1:m
                diff = x_alarm(j) - Zh(i, j);
                dK(i) = -diff * exp(-diff.^2 / (2 * sigma^2)) / sigma^2;
            end
            grad(j) = sum(dK .* (tilde_K - (t * t')));
        end

        contrib_T2 = abs(t.^2) / (sum(abs(t.^2)) + eps0) * sum(t.^2);
        contrib_SPE = abs(grad) / (sum(abs(grad)) + eps0) * abs(SPE_base);
    end

    %=== Print final contribution shapes ===%
    fprintf('[DEBUG] contrib_T2 size: %dx%d | contrib_SPE size: %dx%d\n', ...
        size(contrib_T2,1), size(contrib_T2,2), size(contrib_SPE,1), size(contrib_SPE,2));

end

%=======================================================================
function K = rbf_kernel(X_row, Y, sigma)
    % Computes RBF kernel between one sample X_row and all samples in Y
    D2 = pdist2(X_row, Y, 'euclidean').^2;
    K = exp(-D2 / (2 * sigma^2));
end
