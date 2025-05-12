function [x, y] = positionEstimator(test_data, modelParameters)

    persistent trial_angles;
    
    if isempty(trial_angles)
        trial_angles = containers.Map('KeyType', 'double', 'ValueType', 'double');
    end

    W1 = modelParameters.classifier.W1;
    B1 = modelParameters.classifier.B1;
    W2 = modelParameters.classifier.W2;
    B2 = modelParameters.classifier.B2;
    p = modelParameters.estimator.p;
    delta_mapping = modelParameters.estimator.delta_mapping;

    sigmoid = @(z) 1 ./ (1 + exp(-z));
    
    t = size(test_data.spikes, 2);
    trialId = test_data.trialId;

    if t < 320
        x = test_data.startHandPos(1);
        y = test_data.startHandPos(2);
        return;
    end

    % **Step 1: Classify movement direction only once per trial**
    if ~isKey(trial_angles, trialId)
        X1 = 1/0.001*mean(test_data.spikes(:, 1:300), 2);
        Z1 = W1 * X1 + B1;
        X2 = sigmoid(Z1);
        Z2 = W2 * X2 + B2;
        Y = exp(Z2) ./ sum(exp(Z2));
        [~, predicted_angle] = max(Y);
        
        trial_angles(trialId) = predicted_angle;
    else
        predicted_angle = trial_angles(trialId);
    end

    % **Step 2: Compute Predicted Position Using Δ Mapping**
    t_index = (min(t - 320, 240)) / 240;
    x_poly = polyval(squeeze(p(predicted_angle, 1, :)), t_index);
    y_poly = polyval(squeeze(p(predicted_angle, 2, :)), t_index);

    % Predict Δx, Δy
    delta_x = squeeze(delta_mapping(predicted_angle, 1, :))' * mean(test_data.spikes(:, 1:320), 2);
    delta_y = squeeze(delta_mapping(predicted_angle, 2, :))' * mean(test_data.spikes(:, 1:320), 2);

    % Apply correction
    x = x_poly + delta_x;
    y = y_poly + delta_y;
end
