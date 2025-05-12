function [modelParameters] = positionEstimatorTraining(training_data)
    %% 1) SVM Classifier (unchanged)
    svmTrainDs = {};
    numNeuron = size(training_data(1,1).spikes, 1);  % e.g., 98 neurons
    [numTrial, numDir] = size(training_data);         % e.g., 80 trials, 8 directions
    t_length = 320;  % use first 320 ms for SVM training

    for dir = 1:numDir
        spikeAvg = zeros(numTrial, numNeuron);
        for n = 1:numTrial
            spikeAvg(n, :) = mean(training_data(n,dir).spikes(:,1:t_length),2);
        end
        svmTrainDs{dir} = spikeAvg;
    end

    svmModels = {};
    classes = [1,2,3,4; 5,6,7,8;
               2,3,4,5; 6,7,8,1;
               3,4,5,6; 7,8,1,2;
               4,5,6,7; 8,1,2,3];
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
        y_train_0 = zeros(numTrial*4, 1);
        y_train_1 = ones(numTrial*4, 1);
        y_train = [y_train_0; y_train_1];
        svm = SVM(X_train, y_train, @rbfKernel, 20, 0.01, 500);
        svmModels{numSvm} = svm;
    end
    modelParameters.svmModel = svmModels;
    
    %% 2) Regression for offsets at each time bin
    % Define the time bins (in ms) at which we want to learn offsets.
    timeBins = 320:20:560;   % you can adjust the upper limit if desired
    nBins = length(timeBins);
    
    % Selected neurons (same as before)
    selectedNeurons = [3,7,23,27,28,29,40,41,55,58,61,66,67,68,85,87,88,89,96,98];
    
    % Preallocate cells for average trajectories and regressors.
    avgTraj = cell(1, numDir);  % For each direction, avgTraj{d}.x and .y are vectors (length nBins)
    regressorX = cell(1, numDir);
    regressorY = cell(1, numDir);
    
    % Loop over directions:
    for d = 1:numDir
        % Preallocate matrices for hand positions at each time bin.
        Xd = zeros(numTrial, nBins);
        Yd = zeros(numTrial, nBins);
        for tr = 1:numTrial
            handPos = training_data(tr,d).handPos;  % 2 x T
            for i = 1:nBins
                t_bin = timeBins(i);
                T = size(handPos,2);
                if t_bin > T, t_bin = T; end
                Xd(tr,i) = handPos(1, t_bin);
                Yd(tr,i) = handPos(2, t_bin);
            end
        end
        % Compute the average trajectory (absolute positions) for this direction.
        avgTraj{d}.x = mean(Xd, 1);
        avgTraj{d}.y = mean(Yd, 1);
        
        % For regression targets, we anchor to each trial’s start.
        regX_time = cell(1, nBins);
        regY_time = cell(1, nBins);
        for i = 1:nBins
            targetX = zeros(numTrial, 1);
            targetY = zeros(numTrial, 1);
            for tr = 1:numTrial
                handPos = training_data(tr,d).handPos;
                t_bin = timeBins(i);
                T = size(handPos,2);
                if t_bin > T, t_bin = T; end
                trialOffsetX = handPos(1, t_bin) - handPos(1,1);
                trialOffsetY = handPos(2, t_bin) - handPos(2,1);
                targetX(tr) = trialOffsetX;
                targetY(tr) = trialOffsetY;
            end
            % Compute average start position for this direction.
            startPositions = zeros(numTrial, 2);
            for tr = 1:numTrial
                handPos = training_data(tr,d).handPos;
                startPositions(tr,:) = handPos(1:2,320)'; % using t = 320 as start
            end
            avgStartX = mean(startPositions(:,1));
            avgStartY = mean(startPositions(:,2));
            % Compute average movement offset for this time bin:
            avgMovementX = avgTraj{d}.x(i) - avgStartX;
            avgMovementY = avgTraj{d}.y(i) - avgStartY;
            
            % Set regression target as deviation from average movement.
            targetX = targetX - avgMovementX;
            targetY = targetY - avgMovementY;
            
            % Build the feature matrix F for each trial.
            % --- Updated: use weighted spike counts so that more recent spikes are weighted higher ---
            F = zeros(numTrial, length(selectedNeurons));
            for tr = 1:numTrial
                for nn = 1:length(selectedNeurons)
                    neur = selectedNeurons(nn);
                    % Determine the available time steps (ensure we do not exceed trial length)
                    t_current = min(timeBins(i), size(training_data(tr,d).spikes,2));
                    % Create a weight vector that increases linearly from 1/t_current to 1.
                    weight_vec = (1/t_current : 1/t_current : 1);
                    % Multiply the spikes by the weight vector and normalize by the sum of weights.
                    F(tr, nn) = sum(training_data(tr,d).spikes(neur, 1:t_current) .* weight_vec) / sum(weight_vec);
                end
            end
            % Note: the original code divided by timeBins(i) to compute firing rate.
            % With the weighted approach, each feature is already normalized.
            
            % Train a linear regressor using minimum-norm least-squares.
            beta_x = least_squares_minnorm(F, targetX);
            beta_y = least_squares_minnorm(F, targetY);
            
            regX_time{i} = beta_x;
            regY_time{i} = beta_y;
        end
        regressorX{d} = regX_time;
        regressorY{d} = regY_time;
    end
    
    % Compute the average starting hand position for each direction.
    avgStartPos = cell(1, numDir);
    for d = 1:numDir
        startPositions = zeros(numTrial, 2);
        for tr = 1:numTrial
            startPositions(tr,:) = training_data(tr,d).handPos(1:2,320)';
        end
        avgStartPos{d}.x = mean(startPositions(:,1));
        avgStartPos{d}.y = mean(startPositions(:,2));
    end
    
    %% NEW: Compute average final hand position for each direction
    avgFinalPos = cell(1, numDir);
    for d = 1:numDir
        finalPos = zeros(numTrial, 2);
        for tr = 1:numTrial
            handPos = training_data(tr,d).handPos;  % 2 x T (T varies across trials)
            finalPos(tr,:) = handPos(1:2,end);       % final position of each trial
        end
        avgFinalPos{d}.x = mean(finalPos(:,1));
        avgFinalPos{d}.y = mean(finalPos(:,2));
    end
    
    % Store parameters.
    modelParameters.avgTraj = avgTraj;
    modelParameters.regressorX = regressorX;
    modelParameters.regressorY = regressorY;
    modelParameters.timeBins = timeBins;
    modelParameters.selectedNeurons = selectedNeurons;
    modelParameters.t_start = 320;  % movement onset (for SVM classification)
    modelParameters.avgStartPos = avgStartPos;
    modelParameters.avgFinalPos = avgFinalPos;
    
    % ----------------------------------------
    % Helper: minimum-norm least squares
    % ----------------------------------------
    function w = least_squares_minnorm(A, b)
        lambda = 1.5e-3;  % Regularization parameter (tune as needed)
        [U,S,V] = svd(A, 'econ');
        S_reg = diag(diag(S) ./ (diag(S).^2 + lambda));
        w = V * S_reg * U' * b;
    end
end
