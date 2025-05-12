function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
    % 
    % positionEstimator.m
    % 
    % This version fixes the starburst issue by updating the hand position
    % with dt = 20 ms (the test script's increment), rather than using
    % the windowLength used for feature extraction (e.g., 30 ms).
    %
    % Input:
    %   - test_data: struct with:
    %       * trialId
    %       * startHandPos
    %       * decodedHandPos
    %       * spikes (98 x T matrix of spiking activity)
    %   - modelParameters: struct containing:
    %       * svmModel: 1x4 cell with SVM models
    %       * W_enc, b_enc, W_dec, b_dec: autoencoder weights/biases
    %       * W_reg: 1x8 cell, each containing regression weights for velocity
    %       * direction: integer indicating the last predicted direction
    %
    % Output:
    %   - x, y: the current hand position (in mm)
    %   - newModelParameters: possibly updated version of modelParameters
    %

    % Copy modelParameters so we can modify (e.g., store direction).
    newModelParameters = modelParameters;

    % The competition's test script calls this function at time steps:
    %  320 ms (initial call), then 340 ms, 360 ms, 380 ms, ...
    % We store that time in t_total.
    t_total = size(test_data.spikes, 2);  % current time in ms

    % SVM is used only at t_total == 320 ms to predict direction
    t_length = 320;  % the initial window for SVM direction classification

    if t_total == t_length
        % =============== SVM Direction Prediction ===============
        % Compute average firing rate for the first 320 ms
        X_test = mean(test_data.spikes(:, 1:t_length), 2)';  % 1 x 98

        % Predict direction using the 4 SVM models with a voting scheme
        svm_p = zeros(4, 1);
        for numSvm = 1:4
            pred = SVMPred(modelParameters.svmModel{numSvm}, X_test);
            svm_p(numSvm) = pred;
        end

        % Voting scheme
        classes = [1,2,3,4; 5,6,7,8; 
                   2,3,4,5; 6,7,8,1;
                   3,4,5,6; 7,8,1,2;
                   4,5,6,7; 8,1,2,3];
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

        % Store the predicted direction in newModelParameters
        newModelParameters.direction = y_pred;

        % For the initial call, we haven't updated the position from velocity yet
        % so just return the starting position
        x = test_data.startHandPos(1);
        y = test_data.startHandPos(2);

    else
        % =============== Use Previously Predicted Direction ===============
        % If t_total != 320, we reuse the direction predicted at t=320
        y_pred = newModelParameters.direction;

        % =============== Autoencoder + Ridge Regression for Trajectory ===============
        % We assume you used a 30 ms window for training features, but
        % the test script increments in 20 ms steps after 320 ms. We'll
        % still extract the last 30 ms to compute firing rate, but crucially
        % we only move the hand by velocity * 20 ms each iteration.

        windowLength = 20;  % used for feature extraction
        t_min = t_total - windowLength + 1;  % e.g. if t_total=340, t_min=311

        % Compute mean firing rate over the last 30 ms
        fr = mean(test_data.spikes(:, t_min:t_total), 2);  % 98 x 1

        % Pass through the trained encoder
        h = tanh(newModelParameters.W_enc * fr + newModelParameters.b_enc);  % hidden_dim x 1

        % Retrieve the direction-specific regression weights (hidden_dim x 2)
        W_reg = newModelParameters.W_reg{y_pred};

        % Predict velocity (v_x, v_y). Should be in mm/ms if training was done right.
        v = (h' * W_reg);  % 1 x 2

        % =============== Position Update ===============
        % The test script calls this function every 20 ms after 320 ms,
        % so dt = 20 ms each time we get here.
        dt = 20;  % ms
        decodedHP_len = size(test_data.decodedHandPos, 2);

        % Last decoded position
        x_prev = test_data.decodedHandPos(1, decodedHP_len);
        y_prev = test_data.decodedHandPos(2, decodedHP_len);

        % Update position using velocity * dt
        x = x_prev + v(1) * dt;
        y = y_prev + v(2) * dt;
    end
end
