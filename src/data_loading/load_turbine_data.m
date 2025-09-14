function [healthy_data, faulty1_data, faulty2_data, data_info] = load_turbine_data(filename)
%% load the turbine data from excel file

if nargin < 1
    filename = '../data/data.xlsx';  % default file
end

fprintf('loading wind turbine data...\n');

%% get the data from excel sheets we need
try
    % healthy turbine - need to remove last column to match others
    healthy_table = readtable(filename, 'Sheet', 'No.2WT');
    healthy_data = table2array(healthy_table(:, 1:27));

    % faulty turbines - already have 27 columns
    faulty1_table = readtable(filename, 'Sheet', 'No.14WT');
    faulty1_data = table2array(faulty1_table);

    faulty2_table = readtable(filename, 'Sheet', 'No.39WT');
    faulty2_data = table2array(faulty2_table);

    fprintf('data loaded successfully\n');

catch ME
    error('could not load data: %s', ME.message);
end

%% basic info structure
data_info = struct();
data_info.filename = filename;
data_info.sheets_used = {'No.2WT', 'No.14WT', 'No.39WT'};
data_info.sheets_skipped = {'No.3'};
data_info.final_sizes = [size(healthy_data); size(faulty1_data); size(faulty2_data)];

end