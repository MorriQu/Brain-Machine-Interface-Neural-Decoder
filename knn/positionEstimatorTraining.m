function modelParameters = positionEstimatorTraining(training_data)

    num_trials = size(training_data, 1);
    num_angles = 8;
    num_neurons = 98;  
    num_time_steps = 240;  
    hidden_size = 16;
    epochs = 1000;    
    eth = 0.1;
    m = num_trials * num_angles;
    top_neurons = 20; 

    W1 = randn(hidden_size, num_neurons) * 0.01;
    B1 = zeros(hidden_size, 1);
    W2 = randn(num_angles, hidden_size) * 0.01;
    B2 = zeros(num_angles, 1);

    sigmoid = @(z) 1 ./ (1 + exp(-z));

    PSTH = zeros(num_trials, num_angles, num_neurons);

    % **Find Top 20 Most Active Neurons Per Angle**
    top_neurons_per_angle = zeros(num_angles, top_neurons);

    for k = 1:num_angles
        neural_variance = zeros(num_neurons, 1);
        for n = 1:num_trials
            PSTH(n, k, :) = 1/0.001 * mean(training_data(n, k).spikes(:, 1:300), 2);
            neural_variance = neural_variance + var(training_data(n, k).spikes(:, 1:300), 0, 2);
        end
        [~, sorted_idx] = sort(neural_variance, 'descend');
        top_neurons_per_angle(k, :) = sorted_idx(1:top_neurons);
    end

    % **Train Classifier (MLP)**
    for epoch = 1:epochs
        dW1 = zeros(size(W1));
        dB1 = zeros(size(B1));
        dW2 = zeros(size(W2));
        dB2 = zeros(size(B2));
        total_loss = 0;

        for k = 1:num_angles
            for n = 1:50
                X1 = squeeze(PSTH(n, k, :));
                Yd = zeros(num_angles, 1);
                Yd(k) = 1; 

                Z1 = W1 * X1 + B1;
                X2 = sigmoid(Z1);
                Z2 = W2 * X2 + B2;
                Y = exp(Z2) ./ sum(exp(Z2));

                loss = sum((Y - Yd).^2);
                total_loss = total_loss + loss;

                d2 = (Y - Yd);
                d1 = (W2' * d2) .* (X2 .* (1 - X2));

                dW2 = dW2 + d2 * X2';
                dB2 = dB2 + d2;

                dW1 = dW1 + d1 * X1';
                dB1 = dB1 + d1;
            end
        end

        W2 = W2 - eth/m * dW2;
        B2 = B2 - eth/m * dB2;
        W1 = W1 - eth/m * dW1;
        B1 = B1 - eth/m * dB1;

%         if mod(epoch, 100) == 0
%             fprintf('Epoch %d, Loss: %.4f\n', epoch, total_loss / m);
%         end
    end

    modelParameters.classifier.W1 = W1;
    modelParameters.classifier.B1 = B1;
    modelParameters.classifier.W2 = W2;
    modelParameters.classifier.B2 = B2;
    modelParameters.estimator.top_neurons_per_angle = top_neurons_per_angle;
    modelParameters.estimator.training_data = training_data;
end
