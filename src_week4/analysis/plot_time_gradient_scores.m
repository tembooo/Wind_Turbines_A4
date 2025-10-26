%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Haider Ali
% Plots PC1–PC2 scores with a time gradient per turbine,
% adds arrow from early to late, computes Spearman correlation with time.



function plot_time_gradient_scores(scores, idx, var_explained, title_str)
% Plot PC1–PC2 scores for a given unit block with a 0..1 time gradient.
% - Colors encode time within the unit (early -> late).
% - An arrow shows net drift from early 10% to late 10%.
% - Spearman correlation between time and PC1 is reported in the title.

    % Normalized time within this unit's window (0..1)
    t = linspace(0,1,numel(idx))';
    sxy = scores(idx, 1:2);  % PC1, PC2

    % Scatter with time color
    scatter(sxy(:,1), sxy(:,2), 18, t, 'filled', 'MarkerEdgeColor','none');
    grid on; grid minor; axis tight;
    c = colorbar;
    c.Label.String = 'early \leftarrow  time  \rightarrow late';  % for LaTeX exporter too
    clim([0 1]);

    xlabel(sprintf('PC1 (%.1f%%)', var_explained(1)));
    ylabel(sprintf('PC2 (%.1f%%)', var_explained(2)));

    % Arrow from early-10% centroid to late-10% centroid to show drift
    k = max(1, round(0.10 * numel(idx)));
    p0 = mean(sxy(1:k, :), 1);                 % early centroid
    p1 = mean(sxy(end-k+1:end, :), 1);         % late centroid
    hold on;
    quiver(p0(1), p0(2), p1(1)-p0(1), p1(2)-p0(2), 0, ...
        'k','LineWidth',1.5,'MaxHeadSize',0.7);
    plot(p0(1), p0(2), 'kd', 'MarkerFaceColor','w','MarkerSize',5); % start marker
    plot(p1(1), p1(2), 'ks', 'MarkerFaceColor','w','MarkerSize',5); % end marker
    hold off;

    % Monotonic trend check: Spearman rho between time index and PC1
    n = numel(idx);
    [rho, pval] = corr((1:n)', sxy(:,1), 'type','Spearman');  % time vs PC1
    title(sprintf('%s (\\rho_{time,PC1}=%.2f, p=%.2g)', title_str, rho, pval));
end

