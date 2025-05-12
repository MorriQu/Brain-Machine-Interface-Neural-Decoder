function [x, y] = positionEstimator(test_data, modelParameters)

    persistent trial_angles;
    
    if isempty(trial_angles)
        trial_angles = containers.Map('KeyType', 'double', 'ValueType', 'double');
    end

    W1 = modelParameters.classifier.W1;
    B1 = modelParameters.classifier.B1;
    W2 = modelParameters.classifier.W2;
    B2 = modelParameters.classifier.B2;
    training_data = modelParameters.estimator.training_data;
    
    sigmoid = @(z) 1 ./ (1 + exp(-z));
    
    t = size(test_data.spikes, 2);
    trialId = test_data.trialId;

    x = test_data.startHandPos(1);
    y = test_data.startHandPos(2);

    if ~isKey(trial_angles, trialId)
        X1 = 1/0.001 * mean(test_data.spikes(:, 1:300), 2);
        Z1 = W1 * X1 + B1;
        X2 = sigmoid(Z1);
        Z2 = W2 * X2 + B2;
        Y = exp(Z2) ./ sum(exp(Z2));
        [~, predicted_angle] = max(Y);
        
        trial_angles(trialId) = predicted_angle;
    else
        predicted_angle = trial_angles(trialId);
    end

    if t > 320
        neural_t = test_data.spikes(:, t)';  

        distances = zeros(50, 1);

        for i = 1:50  
            t_train = min(t, 560);  
            train_neural_t = training_data(i, predicted_angle).spikes(:, t_train)'; 

            dot_product = dot(neural_t, train_neural_t);
            norm_product = norm(neural_t) * norm(train_neural_t);
            
            if norm_product == 0
                cosine_similarity = 0;  
            else
                cosine_similarity = dot_product / norm_product;
            end

            distances(i) = 1 - cosine_similarity;  
        end
        
        valid_distances = distances(distances < 1.0);
        k_neighbors = max(11, length(valid_distances));

        [~, sorted_idx] = sort(distances);
        num_neighbors = min(k_neighbors, length(sorted_idx)); 
        nearest_neighbors = sorted_idx(1:num_neighbors);
%         fprintf('k = %d \n', nearest_neighbors);

        t_train = min(t - 320, 240);  

        x = 0;
        y = 0;

        for i = 1:num_neighbors
            trial_idx = nearest_neighbors(i);
            x = x + training_data(trial_idx, predicted_angle).handPos(1, 320 + t_train);
            y = y + training_data(trial_idx, predicted_angle).handPos(2, 320 + t_train);
        end

        x = x / num_neighbors;
        y = y / num_neighbors;
    end
end
