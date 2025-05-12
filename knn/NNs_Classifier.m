clc; clear all; beep off;

% Load Dataset
load('monkeydata_training.mat');

%% Training Classifier

% Initialize Variables
pre = zeros(100, 8, 98); % PSTH matrix
eth = 0.1; % Learning rate
input_size = 98;
hidden_size = 16; 
output_size = 8;  
epochs = 1500;    
m = 50 * 8; % Number of training samples

% Initialize Weights and Biases
W1 = randn(hidden_size, input_size) * 0.01;  
B1 = zeros(hidden_size, 1);                  
W2 = randn(output_size, hidden_size) * 0.01; 
B2 = zeros(output_size, 1);                   

% Activation Functions
sigmoid = @(z) 1 ./ (1 + exp(-z));  
sigmoid_derivative = @(z) z .* (1 - z);

softmax = @(z) exp(z) ./ sum(exp(z));  % Softmax Function

% Creating PSTHs (Feature Extraction)
for n = 1:100
    for k = 1:8
        pre(n, k, :) = 1/0.001 * mean(trial(n, k).spikes(:, 1:300), 2);   
    end    
end

for epoch = 1:epochs
    % Shuffle training order and angles
    train_order = randperm(50);
    angle_order = randperm(8);
    
    % Initialize gradient accumulators
    dW1 = zeros(size(W1));
    dB1 = zeros(size(B1));
    dW2 = zeros(size(W2));
    dB2 = zeros(size(B2));

    total_loss = 0;

    for idx = 1:50
        n = train_order(idx);  % Randomized trial selection

        for j = 1:8
            k = angle_order(j); % Randomized angle selection

            % **1. Prepare Inputs & Target**
            X1 = squeeze(pre(n, k, :)); % Extract 98x1 feature vector
            Yd = zeros(8, 1); % Target vector (8x1)
            Yd(k) = 1; % Set k-th element to 1

            % **2. Forward Propagation**
            Z1 = W1 * X1 + B1; % Hidden layer
            X2 = sigmoid(Z1);

            Z2 = W2 * X2 + B2; % Output layer
            Y = softmax(Z2);  % Apply Softmax

            % **3. Compute Cross-Entropy Loss**
            loss = -sum(Yd .* log(Y));  % Cross-Entropy Loss
            total_loss = total_loss + loss;

            d2 = Y - Yd; 
            d1 = (W2' * d2) .* sigmoid_derivative(X2); 

            % **5. Accumulate Gradients**
            dW2 = dW2 + d2 * X2';
            dB2 = dB2 + d2;

            dW1 = dW1 + d1 * X1';
            dB1 = dB1 + d1;
        end
    end

    % **6. Update Weights and Biases (Batch Update)**
    W2 = W2 - eth/m * dW2;
    B2 = B2 - eth/m * dB2;
    W1 = W1 - eth/m * dW1;
    B1 = B1 - eth/m * dB1;

    % **7. Display Loss Every 100 Epochs**
    if mod(epoch, 100) == 0
        fprintf('Epoch %d, Loss: %.4f\n', epoch, total_loss / m);
    end
end

% **Testing Loop**
correct = 0;
total = 50 * 8;

for n = 51:100
    for k = 1:8
        % **1. Prepare Test Input**
        X1 = squeeze(pre(n, k, :)); % Extract 98x1 feature vector

        % **2. Forward Propagation**
        Z1 = W1 * X1 + B1;
        X2 = sigmoid(Z1);

        Z2 = W2 * X2 + B2;
        Y = softmax(Z2); % Apply Softmax

        % **3. Prediction**
        [~, predicted_class] = max(Y); % Get index of max activation

        % **4. Check Accuracy**
        if predicted_class == k
            correct = correct + 1;
        end
    end
end

% Compute Accuracy
accuracy = (correct / total) * 100;
fprintf('Test Accuracy: %.2f%%\n', accuracy);

