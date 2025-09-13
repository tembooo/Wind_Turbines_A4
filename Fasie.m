%%%%%%%%%%%%%%% 0.25p – Data Import (Fasie Haider)
% Step 1: Load data from Excel file for each turbine
WT2  = readtable('data.xlsx', 'Sheet', 'No.2WT');
WT39 = readtable('data.xlsx', 'Sheet', 'No.39WT');
WT14 = readtable('data.xlsx', 'Sheet', 'No.14WT');
WT3  = readtable('data.xlsx', 'Sheet', 'No.3');

% Step 2: Convert tables to numerical arrays
WT2  = table2array(WT2);
WT39 = table2array(WT39);
WT14 = table2array(WT14);
WT3  = table2array(WT3);

% Step 3: Ensure consistent number of columns across turbines
numColumns = 25;
WT2  = WT2(:,  1:numColumns);
WT39 = WT39(:, 1:numColumns);
WT14 = WT14(:, 1:numColumns);
WT3  = WT3(:,  1:numColumns);

% Step 4: Combine datasets into one matrix
allData = [WT2; WT39; WT14; WT3];

% Display dataset info
[numRecords, numAttributes] = size(allData);
fprintf('Number of records: %d\n', numRecords);
fprintf('Number of attributes: %d\n', numAttributes);


%%%%%%%%%%%%%%% PCA – Fasie Haider
% Step 2: Normalize dataset
allDataNormalized = (allDataClean - mean(allDataClean)) ./ std(allDataClean);

% Step 3: PCA execution
[coeff, score, latent, tsquared, explained] = pca(allDataNormalized);

% Step 5: PCA biplot
figure;
biplot(coeff(:,1:2), 'scores', score(:,1:2));
title('PCA Biplot (PC1 vs PC2)');
