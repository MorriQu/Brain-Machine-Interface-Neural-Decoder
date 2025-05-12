function [x, y, newModelParameters] = positionEstimator(test_data, modelParameters)
    newModelParameters = modelParameters;
    
    t_length = 320;
    X_test = mean(test_data.spikes(:,1:t_length),2)';  % for SVM classification
    t_total = size(test_data.spikes,2);
    
    if t_total > modelParameters.timeBins(end)
        t_total = modelParameters.timeBins(end);
    end

    % --- Anchor at movement onset: if t_total equals 320 ms, return startHandPos.
    if t_total == modelParameters.t_start
        svm_p = zeros(4,1);
        for numSvm = 1:4
            pred = SVMPred(modelParameters.svmModel{numSvm}, X_test);
            svm_p(numSvm) = pred;
        end
        direction = 4 + (svm_p(1)-0.5)*2*sum(svm_p);
        newModelParameters.direction = direction;
        x = test_data.startHandPos(1);
        y = test_data.startHandPos(2);
        return;
    end

    % --- SVM Classification (if t_total == 320 already handled above)
    if t_total == t_length
        svm_p = zeros(4,1);
        for numSvm = 1:4
            pred = SVMPred(modelParameters.svmModel{numSvm}, X_test);
            svm_p(numSvm) = pred;
        end
        direction = 4 + (svm_p(1)-0.5)*2*sum(svm_p);
        newModelParameters.direction = direction;
    else
        direction = newModelParameters.direction;
    end

    if t_total < modelParameters.t_start
        x = test_data.startHandPos(1);
        y = test_data.startHandPos(2);
        return;
    end

    timeBins = modelParameters.timeBins;
    idx = find(timeBins <= t_total, 1, 'last');
    if isempty(idx)
        idx = 1;
    end

    avgX = modelParameters.avgTraj{direction}.x;
    avgY = modelParameters.avgTraj{direction}.y;
    
    % --- Updated: Compute weighted firing rate features for test data ---
    selectedNeurons = modelParameters.selectedNeurons;
    F_test = zeros(1, length(selectedNeurons));
    % Recalculate the weight vector: linearly increasing from 1/t_total to 1.
    weight_vec = (1/t_total : 1/t_total : 1);
    for iN = 1:length(selectedNeurons)
        neur = selectedNeurons(iN);
        F_test(iN) = sum(test_data.spikes(neur,1:t_total) .* weight_vec) / sum(weight_vec);
    end
    
    % Predict the offset using the linear regressor trained for this bin.
    beta_x = modelParameters.regressorX{direction}{idx};
    beta_y = modelParameters.regressorY{direction}{idx};
    predOffsetX = F_test * beta_x;
    predOffsetY = F_test * beta_y;
    
    % Incorporate anchoring using training average movement offset.
    avgStartX = modelParameters.avgStartPos{direction}.x;
    avgStartY = modelParameters.avgStartPos{direction}.y;
    avgMovementX = avgX(idx) - avgStartX;
    avgMovementY = avgY(idx) - avgStartY;
    
    % Update position based on regression offset or final average position.
    if t_total < modelParameters.timeBins(end)
        gamma = 0.67;
        finalX = test_data.startHandPos(1) + avgMovementX + gamma * predOffsetX;
        finalY = test_data.startHandPos(2) + avgMovementY + gamma * predOffsetY;
    else
        gamma = 0.9;
        currentX = test_data.startHandPos(1) + avgMovementX + gamma * predOffsetX;
        currentY = test_data.startHandPos(2) + avgMovementY + gamma * predOffsetY;
        alpha = 0.82; % Tunable velocity factor (adjust as needed)
        finalX = currentX + alpha * (modelParameters.avgFinalPos{direction}.x - currentX);
        finalY = currentY + alpha * (modelParameters.avgFinalPos{direction}.y - currentY);
    end
    
    x = finalX;
    y = finalY;
end