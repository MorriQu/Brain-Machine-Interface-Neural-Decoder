function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
newModelParameters = modelParameters; 
%% SVM predicts movement direction
    t_length = 320;  % Time window for SVM prediction is 320 ms
    X_test = mean(test_data.spikes(:, 1:t_length), 2)';  % Convert the average firing rate of each neuron over 320 ms to a row vector
    t_total = size(test_data.spikes, 2);  % Total duration of current test_data
    
    if t_total == 320
        % If the data is the initial 320 ms, use the SVM model to predict the direction
        svm_p = zeros(4, 1);
        for numSvm = 1:4
            pred = SVMPred(modelParameters.svmModel{numSvm}, X_test);
            svm_p(numSvm) = pred;
        end
        
        % Determine the direction using a voting method based on the prediction results from each SVM model
        classes = [1,2,3,4; 5,6,7,8; 2,3,4,5; 6,7,8,1; 3,4,5,6; 7,8,1,2; 4,5,6,7; 8,1,2,3];
        votes = zeros(8, 1);
        for numSvm = 1:4
            if svm_p(numSvm) == 1
                dirs = classes(2*numSvm, :);
            else
                dirs = classes(2*numSvm - 1, :);
            end
            votes(dirs) = votes(dirs) + 1;
        end
        [~, y_pred] = max(votes);
        newModelParameters.direction = y_pred;
    else
        % If it is not the initial 320 ms, then directly use the previously predicted direction
        y_pred = newModelParameters.direction;
    end

    %% Trajectory prediction using linear regression
    % Obtain the current predicted direction and the selected important neurons
    direction = newModelParameters.direction;
    selected_neurons = newModelParameters.selectedNeurons;
    
    t_lag = 20;  % Time window for calculating the firing rate (20 ms)
    t_min = t_total - t_lag + 1;  % Starting index of the current time window
    
    % Calculate the firing rate of the selected neurons over the last 20 ms
    fRates = zeros(length(selected_neurons), 1);
    for n = 1:length(selected_neurons)
        spike_segment = test_data.spikes(selected_neurons(n), t_min:t_total);
        fRates(n) = sum(spike_segment) / t_lag;
    end
    
    % Predict hand velocity (v_x, v_y) using the regression parameters obtained during training
    regres_coeff = newModelParameters.regres{direction};
    v_x = fRates' * regres_coeff(:, 1);
    v_y = fRates' * regres_coeff(:, 2);
    
    % Update the hand position based on the predicted velocity
    if t_total ~= 320
        % For non-initial states: update the current position using the last decoded hand position plus the displacement
        decodedHP_len = size(test_data.decodedHandPos, 2);
        x = test_data.decodedHandPos(1, decodedHP_len) + v_x * t_lag;
        y = test_data.decodedHandPos(2, decodedHP_len) + v_y * t_lag;
    else
        % For the initial state: directly use the starting hand position
        x = test_data.startHandPos(1);
        y = test_data.startHandPos(2);
    end
end
