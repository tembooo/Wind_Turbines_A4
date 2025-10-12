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
