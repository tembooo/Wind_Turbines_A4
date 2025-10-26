%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Haider Ali
% Computes scores for new data in kPCA.
% Mathematical description: \tilde{K}_{new} = K_{new} - 1_{m,n} K - K_{new} 1_n + 1_{m,n} K 1_n
%   scores = \tilde{K}_{new} alpha

function scores = compute_scores_kpca(Z_new, kernel_info, alpha)
% Project new data onto k-PCA model

    sigma = kernel_info.sigma;
    Zh = kernel_info.Zh;
    nh = size(Zh,1);
    K_new = rbf_kernel(Z_new, Zh, sigma);

    m = size(Z_new,1);
    one_n_t = ones(m, nh) / nh;
    one_n = kernel_info.one_n;

    tilde_K_new = K_new - one_n_t * kernel_info.K - K_new * one_n + one_n_t * kernel_info.K * one_n;

    scores = tilde_K_new * alpha;
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
