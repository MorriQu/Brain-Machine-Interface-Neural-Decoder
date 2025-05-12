function modelParameters = positionEstimatorTraining(training_data)

    num_trials = size(training_data, 1);
    num_angles = 8;
    num_neurons = 98;  
    num_time_steps = 240;  
    hidden_size = 64;
    epochs = 1000;    
    eth = 0.1;
    poly_degree = 6;
    m = num_trials * num_angles;  

    % Initialize network parameters
    W1 = randn(hidden_size, num_neurons) * 0.01;
    B1 = zeros(hidden_size, 1);
    W2 = randn(num_angles, hidden_size) * 0.01;
    B2 = zeros(num_angles, 1);

    sigmoid = @(z) 1 ./ (1 + exp(-z));
    sigmoid_derivative = @(z) z .* (1 - z);

    PSTH = zeros(num_trials, num_angles, num_neurons);
    for n = 1:num_trials
        for k = 1:num_angles
            PSTH(n, k, :) = 1/0.001 * mean(training_data(n, k).spikes(:, 1:300), 2);
        end
    end

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
                d1 = (W2' * d2) .* sigmoid_derivative(X2);

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

        if mod(epoch, 100) == 0
            fprintf('Epoch %d, Loss: %.4f\n', epoch, total_loss / m);
        end
    end

    % **Step 1: Compute Mean Trajectories Over All Training Trials**  
    pos_mean = zeros(num_angles, 2, num_time_steps);

    for k = 1:num_angles
        pos_sum = zeros(2, num_time_steps);
        for n = 1:50
            pos_sum = pos_sum + training_data(n, k).handPos(1:2, 321:560);
        end
        pos_mean(k, :, :) = pos_sum / 50;
    end

    % **Step 2: Fit Polynomial and Compute Δ Mapping**  
    p = zeros(num_angles, 2, poly_degree + 1);
    delta_mapping = zeros(num_angles, 2, num_neurons); 

    t = (1:num_time_steps) / num_time_steps;

    for k = 1:num_angles
        delta_train = zeros(50, 2);
        N_avg_train = zeros(50, num_neurons);

        for n = 1:50
            x_trial = training_data(n, k).handPos(1, 321:560);
            y_trial = training_data(n, k).handPos(2, 321:560);

            delta_train(n, :) = [mean(x_trial - squeeze(pos_mean(k, 1, :))'), ...
                                 mean(y_trial - squeeze(pos_mean(k, 2, :))')];

            N_avg_train(n, :) = mean(training_data(n, k).spikes(:, 1:320), 2);
        end

        % Fit polynomial for trajectory
        p(k, 1, :) = polyfit(t, squeeze(pos_mean(k, 1, :)), poly_degree);
        p(k, 2, :) = polyfit(t, squeeze(pos_mean(k, 2, :)), poly_degree);

        % Compute Δ mapping using least squares regression
        delta_mapping(k, :, :) = delta_train' * pinv(N_avg_train');
    end

    % Store trained model parameters
    modelParameters.classifier.W1 = W1;
    modelParameters.classifier.B1 = B1;
    modelParameters.classifier.W2 = W2;
    modelParameters.classifier.B2 = B2;
    modelParameters.estimator.p = p;
    modelParameters.estimator.delta_mapping = delta_mapping;
end
