function K = rbf_kernel(X1, X2, sigma)
% RBF kernel: exp(-||x1 - x2||^2 / (2 sigma^2))
    n1 = size(X1,1);
    n2 = size(X2,1);
    norm1 = sum(X1.^2,2);
    norm2 = sum(X2.^2,2);
    dist = repmat(norm1,1,n2) + repmat(norm2',n1,1) - 2 * X1 * X2';
    K = exp( - dist / (2 * sigma^2 ) );
end