# hansel.github.io
%Bearing health classification machine learning algorithm

% This code was created on matlab aiming to classify different classes of bearing health.
% The classes are:
% 1. Healthy bearing
% 2. Inner race defect
% 3. Outer race defect
% 4. Scratched ball defect
% 5. Combination of all defects.

%The aim is to achieve an accuracy of at least 80%.
%The model could be useful in industrial envrionments where bearing vibrations can be monitored and replaced ahead of its end of life.
%Changing bearings early can prevent inefficiencies in machines and prevent more expensive components from failing.

The code starts beneath the following line
-------------------------------------------------------------------------------------------------------------------------------------------
% Set up parallel pool (if you have parallel computing toolbox)
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool;  % Create a parallel pool if not already running
end

% Load .mat files
HA1 = load('HEALTHY_BEARING/H-A-1.mat');
HA2 = load('HEALTHY_BEARING/H-A-2.mat');
HA3 = load('HEALTHY_BEARING/H-A-3.mat');
IA1 = load('INNER_RACE/I-A-1.mat');
IA2 = load('INNER_RACE/I-A-2.mat');
IA3 = load('INNER_RACE/I-A-3.mat');
OA1 = load('OUTER_RACE/O-A-1.mat');
OA2 = load('OUTER_RACE/O-A-2.mat');
OA3 = load('OUTER_RACE/O-A-3.mat');
BA1 = load('SCRATCHED_BALL/B-A-1.mat');
BA2 = load('SCRATCHED_BALL/B-A-2.mat');
BA3 = load('SCRATCHED_BALL/B-A-3.mat');


% Convert to tables
HA1 = array2table([HA1.Channel_1, HA1.Channel_2], 'VariableNames', {'Vibration', 'Speed'});
HA2 = array2table([HA2.Channel_1, HA2.Channel_2], 'VariableNames', {'Vibration', 'Speed'});
HA3 = array2table([HA3.Channel_1, HA3.Channel_2], 'VariableNames', {'Vibration', 'Speed'});
IA1 = array2table([IA1.Channel_1, IA1.Channel_2], 'VariableNames', {'Vibration', 'Speed'});
IA2 = array2table([IA2.Channel_1, IA2.Channel_2], 'VariableNames', {'Vibration', 'Speed'});
IA3 = array2table([IA3.Channel_1, IA3.Channel_2], 'VariableNames', {'Vibration', 'Speed'});
OA1 = array2table([OA1.Channel_1, OA1.Channel_2], 'VariableNames', {'Vibration', 'Speed'});
OA2 = array2table([OA2.Channel_1, OA2.Channel_2], 'VariableNames', {'Vibration', 'Speed'});
OA3 = array2table([OA3.Channel_1, OA3.Channel_2], 'VariableNames', {'Vibration', 'Speed'});
BA1 = array2table([BA1.Channel_1, BA1.Channel_2], 'VariableNames', {'Vibration', 'Speed'});
BA2 = array2table([BA2.Channel_1, BA2.Channel_2], 'VariableNames', {'Vibration', 'Speed'});
BA3 = array2table([BA3.Channel_1, BA3.Channel_2], 'VariableNames', {'Vibration', 'Speed'});


% Function to extract features from each data point
function features = extract_features(data)
    % Calculate time-domain features for each data point
    mean_val = mean(data);
    std_val = std(data);
    skew_val = skewness(data);
    kurt_val = kurtosis(data);
    
    % FFT features (frequency-domain)
    n = length(data);
    fft_vals = abs(fft(data));
    fft_mean = mean(fft_vals);
    fft_max = max(fft_vals);
    
    % Combine features into a single row
    features = [mean_val, std_val, skew_val, kurt_val, fft_mean, fft_max];
end

% Helper function to process the dataset in batches
function [features, lbls] = process_dataset_in_batches(dataset, fault_label, batch_size)
    % Initialize feature array
    features = [];
    lbls = [];
    
    % Process the dataset in batches
    num_batches = ceil(height(dataset) / batch_size);
    
    for batch = 1:num_batches
        start_idx = (batch - 1) * batch_size + 1;
        end_idx = min(batch * batch_size, height(dataset));
        
        % Extract the data for the current batch
        batch_data = dataset(start_idx:end_idx, :); % Slice the data
        
        % Extract the vibration data (convert to numeric array)
        vibration_data = table2array(batch_data(:, 'Vibration')); % Convert to numeric array
        
        % Extract features for the current batch
        batch_features = arrayfun(@(x) extract_features(vibration_data(x, :)), (1:size(vibration_data, 1))', 'UniformOutput', false);
        
        % Concatenate the batch features
        features = [features; cell2mat(batch_features)];
        
        % Assign fault labels to the current batch
        batch_labels = repmat(fault_label, size(batch_features, 1), 1);
        lbls = [lbls; batch_labels];
        
        % Save incrementally to avoid memory overload
        if mod(batch, 5) == 0  % Save every 5 batches
            save('features_in_progress.mat', 'features', 'lbls');
        end
    end
end

% Define batch size
batch_size = 10000;

% Process each dataset in batches
[features_HA1, labels_HA1] = process_dataset_in_batches(HA1, 0, batch_size);
[features_HA2, labels_HA2] = process_dataset_in_batches(HA2, 0, batch_size);
[features_HA3, labels_HA3] = process_dataset_in_batches(HA3, 0, batch_size);

[features_IA1, labels_IA1] = process_dataset_in_batches(IA1, 1, batch_size);
[features_IA2, labels_IA2] = process_dataset_in_batches(IA2, 1, batch_size);
[features_IA3, labels_IA3] = process_dataset_in_batches(IA3, 1, batch_size);

[features_OA1, labels_OA1] = process_dataset_in_batches(OA1, 2, batch_size);
[features_OA2, labels_OA2] = process_dataset_in_batches(OA2, 2, batch_size);
[features_OA3, labels_OA3] = process_dataset_in_batches(OA3, 2, batch_size);

[features_BA1, labels_BA1] = process_dataset_in_batches(BA1, 3, batch_size);
[features_BA2, labels_BA2] = process_dataset_in_batches(BA2, 3, batch_size);
[features_BA3, labels_BA3] = process_dataset_in_batches(BA3, 3, batch_size);

% Concatenate all features and labels
all_features = [
    features_HA1;
    features_HA2;
    features_HA3;
    features_IA1;
    features_IA2;
    features_IA3;
    features_OA1;
    features_OA2;
    features_OA3;
    features_BA1;
    features_BA2;
    features_BA3
];

all_labels = [
    labels_HA1;
    labels_HA2;
    labels_HA3;
    labels_IA1;
    labels_IA2;
    labels_IA3;
    labels_OA1;
    labels_OA2;
    labels_OA3;
    labels_BA1;
    labels_BA2;
    labels_BA3
];

% Convert to table
features_table = array2table(all_features, 'VariableNames', {'Mean', 'StdDev', 'Skewness', 'Kurtosis', 'FFT_Mean', 'FFT_Max'});
features_table.Fault = all_labels;  % Add fault labels

% Save final features table
save('final_features.mat', 'features_table');

% 2D Scatter Plot (Mean vs Standard Deviation)
figure;
gscatter(features_table.Mean, features_table.StdDev, features_table.Fault, 'rgbc', 'o^s', 8);
xlabel('Mean');
ylabel('Standard Deviation');
title('2D Scatter Plot of Mean vs Standard Deviation');
legend('Healthy', 'Inner Race Fault', 'Outer Race Fault', 'Ball Bearing Fault');
grid on;
