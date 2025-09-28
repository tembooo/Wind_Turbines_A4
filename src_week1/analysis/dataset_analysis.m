%% wind turbine dataset analysis script
% shows dimensions

clear; clc; close all;

%% load data using our reusable function
fprintf('=== dataset dimension analysis ===\n\n');

% original dimensions before preprocessing
sheets = {'No.2WT', 'No.3', 'No.14WT', 'No.39WT'};
turbine_names = {'WT2 (healthy)', 'WT3 (faulty)', 'WT14 (faulty)', 'WT39 (faulty)'};
dimensions = zeros(length(sheets), 2);

fprintf('original data dimensions:\n');
for i = 1:length(sheets)
    try
        data = readtable('../data/data.xlsx', 'Sheet', sheets{i});
        dimensions(i,1) = size(data, 1);
        dimensions(i,2) = size(data, 2);
        fprintf('  %s: %d rows x %d cols\n', sheets{i}, dimensions(i,1), dimensions(i,2));
    catch ME
        fprintf('  error reading %s: %s\n', sheets{i}, ME.message);
        dimensions(i,:) = [NaN, NaN];
    end
end

%load preprocessed data
fprintf('\nloading preprocessed data...\n');
addpath('../data_loading');  % add path to data loading functions
try
    [healthy_matrix, faulty1_matrix, faulty2_matrix, data_info] = load_turbine_data();
catch ME
    fprintf('error loading data: %s\n', ME.message);
    return;
end


%% summary tables
fprintf('\n=== original data summary ===\n');
fprintf('%-12s %-15s %8s %8s\n', 'sheet', 'turbine', 'rows', 'cols');
fprintf('%-12s %-15s %8s %8s\n', '-----', '-------', '----', '----');
for i = 1:length(sheets)
    if ~isnan(dimensions(i,1))
        fprintf('%-12s %-15s %8d %8d\n', sheets{i}, turbine_names{i}, ...
                dimensions(i,1), dimensions(i,2));
    end
end

fprintf('\n=== final processed data ===\n');
fprintf('%-12s %-15s %8s %8s\n', 'dataset', 'turbine', 'rows', 'cols');
fprintf('%-12s %-15s %8s %8s\n', '-------', '-------', '----', '----');
fprintf('%-12s %-15s %8d %8d\n', 'healthy', 'WT2 (processed)', size(healthy_matrix,1), size(healthy_matrix,2));
fprintf('%-12s %-15s %8d %8d\n', 'faulty1', 'WT14', size(faulty1_matrix,1), size(faulty1_matrix,2));
fprintf('%-12s %-15s %8d %8d\n', 'faulty2', 'WT39', size(faulty2_matrix,1), size(faulty2_matrix,2));