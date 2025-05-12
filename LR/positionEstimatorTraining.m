function [modelParameters, firingRates, velocities] = positionEstimatorTraining(training_data)
    t_length = 320;   % Spike data for the first 320 ms before movement
    t_lag = 20;       % Time window of 20 ms
    t_max = 570;      % Maximum trajectory length

    %% SVM Classifier Section
    svmTrainDs = cell(1,8); % Store features for each direction
    
    numNeuron = size(training_data(1, 1).spikes, 1); % For example, 98 neurons
    [numTrial, numDir] = size(training_data);         % For example, 80 trials, 8 directions

    % For each direction, calculate the average firing rate of each neuron over 320 ms
    for dir = 1:numDir
        spikeAvg = zeros(numTrial, numNeuron);
        for n = 1:numTrial
            spikeAvg(n,:) = mean(training_data(n,dir).spikes(:,1:t_length), 2);
        end
        svmTrainDs{dir} = spikeAvg;
    end

    svmModels = cell(1,4);
    classes = [1,2,3,4; 5,6,7,8;
               2,3,4,5; 6,7,8,1;
               3,4,5,6; 7,8,1,2;
               4,5,6,7; 8,1,2,3];

    sigma = 0.1; % RBF kernel parameter
    for numSvm = 1:4
        svmTrainDs_0 = [];
        svmTrainDs_1 = [];
        dirs_0 = classes(2*numSvm - 1, :);
        dirs_1 = classes(2*numSvm, :);
        for k = 1:4
            svmTrainDs_0 = [svmTrainDs_0, svmTrainDs{dirs_0(k)}'];
            svmTrainDs_1 = [svmTrainDs_1, svmTrainDs{dirs_1(k)}'];
        end
        svmTrainDs_0 = svmTrainDs_0';
        svmTrainDs_1 = svmTrainDs_1';
        X_train = [svmTrainDs_0; svmTrainDs_1];
        y_train_0 = zeros(numTrial*4,1);
        y_train_1 = ones(numTrial*4,1);
        y_train = [y_train_0; y_train_1];
        svm = SVM(X_train, y_train, @(X1,X2) rbfKernel(X1,X2,sigma), 20, 0.01, 500);
        svmModels{numSvm} = svm;
    end
    modelParameters.svmModel = svmModels;

    %% Linear Regression Section: Automatically Select the Most Relevant Neurons
    all_firing = [];
    all_vel_x = [];
    all_vel_y = [];
    for dir = 1:numDir
        for tr = 1:numTrial
            t_bins = t_length:t_lag:(t_max - t_lag);
            for t = t_bins
                % Calculate the average firing rate of all neurons over the current time window
                fRates_all = mean(training_data(tr,dir).spikes(:, t:t+t_lag), 2);
                all_firing = [all_firing, fRates_all];
                
                % Compute the speed (x component) at the current time
                x = training_data(tr,dir).handPos(1,t);
                x_next = training_data(tr,dir).handPos(1,t+t_lag);
                vel_x = (x_next - x) / t_lag;
                all_vel_x = [all_vel_x, vel_x];
                
                % Compute the speed (y component) at the current time
                y = training_data(tr,dir).handPos(2,t);
                y_next = training_data(tr,dir).handPos(2,t+t_lag);
                vel_y = (y_next - y) / t_lag;
                all_vel_y = [all_vel_y, vel_y];
            end
        end
    end

    % Compute the correlation of each neuron's firing rate with the hand velocity (importance)
    importance = zeros(numNeuron,1);
    for n = 1:numNeuron
        corr_x = corr(all_firing(n,:)', all_vel_x');
        corr_y = corr(all_firing(n,:)', all_vel_y');
        importance(n) = (abs(corr_x) + abs(corr_y)) / 2;
    end

    % Select the top 20 most important neurons
    [~, sorted_idx] = sort(importance, 'descend');
    selected_neurons = sorted_idx(1:20);
    modelParameters.selectedNeurons = selected_neurons;

    %% Linear Regression Training (Using Ridge Regression)
    regres = cell(1, numDir);
    firingRates = cell(length(selected_neurons), numDir);
    velocities = cell(1, numDir);
    for dir = 1:numDir
        firingR = [];
        vel_x_all = [];
        vel_y_all = [];
        for tr = 1:numTrial
            t_bins = t_length:t_lag:(t_max - t_lag);
            for t = t_bins
                % Compute the average firing rate for the selected neurons over the current time window
                fr = mean(training_data(tr,dir).spikes(selected_neurons, t:t+t_lag), 2)';
                firingR = [firingR; fr];
                % Compute the corresponding speed (x component)
                x = training_data(tr,dir).handPos(1,t);
                x_next = training_data(tr,dir).handPos(1,t+t_lag);
                vel_x = (x_next - x) / t_lag;
                vel_x_all = [vel_x_all; vel_x];
                % Compute the corresponding speed (y component)
                y = training_data(tr,dir).handPos(2,t);
                y_next = training_data(tr,dir).handPos(2,t+t_lag);
                vel_y = (y_next - y) / t_lag;
                vel_y_all = [vel_y_all; vel_y];
            end
        end
        velocity = [vel_x_all, vel_y_all];
        regres{dir} = ridge_regression(firingR, velocity, 0.1);
        firingRates(:,dir) = num2cell(firingR, 1);
        velocities{dir}.x = vel_x_all;
        velocities{dir}.y = vel_y_all;
    end
    modelParameters.regres = regres;
end

% Ridge Regression Function (Least Squares with Regularization)
function w = ridge_regression(A, b, lambda)
    w = (A' * A + lambda * eye(size(A,2))) \ (A' * b);
end
