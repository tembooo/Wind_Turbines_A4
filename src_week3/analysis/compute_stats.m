%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Haider Ali
% Computes T2 and SPE for PCA and kPCA models.
% For PCA, parametric limits (F-dist for T2, chi^2 approx for SPE).
% For kPCA, empirical quantiles (95th percentile on healthy).
% Mathematical description: PCA T2_i = t_i^T Λ_a^{-1} t_i (Hotelling's, F(nh,a,nh-a))
%   SPE_i = ||e_i||^2 = ||x_i - \hat{x}_i||^2, limit via moments theta1/2/3
%   kPCA T2_i = sum(t_{i,k}^2), SPE_i = \tilde{k}_{ii} - t_i^T t_i (kernel residual)
%   Limits: prctile on healthy stats.

function [T2, SPE, T2_limit, SPE_limit] = compute_stats(model_type, scores, lambda, alpha_level, nh, Z, hat_Z, kernel_info)
    if strcmp(model_type, 'pca')
        a = length(lambda);
        T2 = sum( (scores .^2) ./ lambda', 2 ); % Vectorized T2
        E = Z - hat_Z;
        SPE = sum(E.^2, 2);
        % Parametric limits
        T2_limit = (a * (nh^2 - 1)) / (nh * (nh - a)) * finv(1 - alpha_level, a, nh - a);
        theta1 = sum(lambda(a+1:end));
        theta2 = sum(lambda(a+1:end).^2);
        theta3 = sum(lambda(a+1:end).^3);
        h0 = 1 - 2 * theta1 * theta3 / (3 * theta2^2);
        if h0 <= 0, h0 = 0.001; end
        ca = norminv(1 - alpha_level);
        SPE_limit = theta1 * (ca * sqrt(2 * theta2 * h0^2)/theta1 + 1 + theta2 * h0 * (h0 - 1)/theta1^2)^(1/h0);
    elseif strcmp(model_type, 'kpca')
        T2 = sum(scores.^2, 2); % Simple sum (no lambda normalization in basic kPCA T2)
        n = size(Z,1);
        SPE = zeros(n,1);
        sigma = kernel_info.sigma;
        Zh = kernel_info.Zh;
        mean_train = kernel_info.mean_train;
        K_all = rbf_kernel(Z, Zh, sigma);
        mean_k = mean(K_all, 2); % Mean kernel per row
        SPE = 1 - 2 * mean_k + mean_train - sum(scores.^2, 2); % Kernel SPE
        % Empirical limits on healthy
        T2_limit = [];
        SPE_limit = [];
        if nargin > 4 && ~isempty(nh) && n >= nh  % Only compute limits if full healthy data is provided
            T2_h = T2(1:nh);
            SPE_h = SPE(1:nh);
            T2_limit = prctile(T2_h, 100*(1 - alpha_level));
            SPE_limit = prctile(SPE_h, 100*(1 - alpha_level));
        end
    else
        error('Unknown model_type: %s', model_type);
    end
end
