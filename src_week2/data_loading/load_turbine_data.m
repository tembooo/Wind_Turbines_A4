%% ===================== Local functions (each defined once) =====================
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Arman Golbidi
% This function loads the turbine datasets from Excel (Healthy, Faulty1, Faulty2),
% removes numeric header rows if present, and returns data matrices + info struct.



function [healthy_data, faulty1_data, faulty2_data, data_info] = load_turbine_data(filename)
% Load the turbine data from Excel file into numeric arrays.
% Removes a numeric label/header row at the top if detected (row-wise labels or numeric header).
% Returns three data matrices and an info struct.
    if nargin < 1
        filename = '../data/data.xlsx';  % default file
    end
    fprintf('loading wind turbine data...\n');
    try
        % Healthy WT (keep first 27 vars to match others)
        healthy_table = readtable(filename, 'Sheet', 'No.2WT');
        Xh = table2array(healthy_table(:, 1:27));
        [Xh, dropped_h_row] = drop_numeric_header_row(Xh, 'No.2WT');

        % Faulty WT #1 (27 vars)
        faulty1_table = readtable(filename, 'Sheet', 'No.14WT');
        X1 = table2array(faulty1_table);
        [X1, dropped_f1_row] = drop_numeric_header_row(X1, 'No.14WT');

        % Faulty WT #2 (27 vars)
        faulty2_table = readtable(filename, 'Sheet', 'No.39WT');
        X2 = table2array(faulty2_table);
        [X2, dropped_f2_row] = drop_numeric_header_row(X2, 'No.39WT');

        healthy_data = Xh;
        faulty1_data = X1;
        faulty2_data = X2;

        fprintf('data loaded successfully');
        if dropped_h_row || dropped_f1_row || dropped_f2_row
            fprintf(' (numeric top-row header/labels removed where detected)');
        end
        fprintf('\n');

    catch ME
        error('could not load data: %s', ME.message);
    end

    % info struct
    data_info = struct();
    data_info.filename       = filename;
    data_info.sheets_used    = {'No.2WT', 'No.14WT', 'No.39WT'};
    data_info.sheets_skipped = {'No.3'};
    data_info.final_sizes    = [size(healthy_data); size(faulty1_data); size(faulty2_data)];

    % Optional variable names (from healthy sheet)
    if width(healthy_table) >= 27
        data_info.var_names = healthy_table.Properties.VariableNames(1:27);
    else
        data_info.var_names = {};
    end
end
