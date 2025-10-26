%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Fasie Haider
% Computes variable contributions to T2 and SPE for PCA and kPCA models.

function [contrib_T2, contrib_SPE] = compute_contributions(model_type, alarm_idx, Z, scores, latent, coeffs, hat_Z, kernel_info, alpha)

    %=== Check alarm index validity ===%
    if isempty(alarm_idx) || alarm_idx > size(Z, 1)
        error('Invalid alarm index');
    end

    n_vars = size(Z, 2);
    contrib_T2 = zeros(1, n_vars);
    contrib_SPE = zeros(1, n_vars);

    eps0 = eps; % small constant for numerical stability

    if strcmp(model_type, 'pca')
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
        %===== kPCA branch =====%
        if isempty(kernel_info)
            error('kernel_info required for kPCA contributions');
        end
        if isempty(alpha)
            error('alpha required for kPCA contributions');
        end

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
            error('alpha shape incompatible with Zh (%d vs %d)', m, ra);
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


end
















%=======================================================================
function K = rbf_kernel(X_row, Y, sigma)
    % Computes RBF (Gaussian) kernel between one sample X_row and all samples in Y
    D2 = pdist2(X_row, Y, 'euclidean').^2;
    K = exp(-D2 / (2 * sigma^2));
end
