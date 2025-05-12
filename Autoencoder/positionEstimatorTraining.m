function [modelParameters] = positionEstimatorTraining(training_data)
    %% Timing Parameters (kept as in the original model)
    t_length = 320;   % initial 320 ms used for SVM prediction
    % For the sliding window approach, we now use:
    windowLength = 20;   % window length in ms (new)
    stepSize = 20;       % step size in ms (overlap: 30ms window moves every 10ms)
    t_max = 570;         % maximum trajectory length (ms)
    
    %% 1. SVM Classifier Section (unchanged)
    numNeuron = size(training_data(1, 1).spikes, 1);  % should be 98 neurons
    [numTrial, numDir] = size(training_data);         % e.g., 100 trials x 8 directions
    
    svmTrainDs = cell(1, numDir);
    % For each direction, compute the average firing rate over the first 320 ms
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
    sigma = 0.1;  % RBF kernel parameter
    
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
    
    %% 2. Autoencoder + Ridge Regression for Trajectory Estimation
    % In this new design, we extract overlapping sliding window features from all 98 neurons.
    % Each feature vector is 98-dimensional (mean firing rate over a 30 ms window).
    
    X_all = [];      % Each row: 98-dim feature vector for one 30ms window
    Y_all = [];      % Each row: corresponding hand velocity [vel_x, vel_y]
    dir_labels = []; % Movement direction label (1 to 8) for each sample
    
    for dir = 1:numDir
        for tr = 1:numTrial
            % Create overlapping windows: from t_length to (t_max - windowLength + 1)
            t_bins = t_length : stepSize : (t_max - windowLength + 1);
            for t = t_bins
                % Extract mean firing rate from all 98 neurons over current 30ms window
                fr = mean(training_data(tr,dir).spikes(:, t : t+windowLength-1), 2)';  % 1 x 98 vector
                X_all = [X_all; fr];
                
                % Compute hand velocity (x component)
                x     = training_data(tr,dir).handPos(1,t);
                xNext = training_data(tr,dir).handPos(1,t+windowLength-1);
                vel_x = (xNext - x) / windowLength;
                
                % Compute hand velocity (y component)
                y     = training_data(tr,dir).handPos(2,t);
                yNext = training_data(tr,dir).handPos(2,t+windowLength-1);
                vel_y = (yNext - y) / windowLength;
                Y_all = [Y_all; [vel_x, vel_y]];
                
                % Record direction label
                dir_labels = [dir_labels; dir];
            end
        end
    end
    
    %% 2b. Train a simple autoencoder on X_all.
    % The autoencoder will learn to compress the 98-dimensional input to a lower-dimensional latent space.
    input_dim = size(X_all,2);  % should be 98
    hidden_dim = 30;            % latent space dimension (tunable)
    learning_rate = 0.5;
    num_iter = 1000;
    
    % Initialize encoder weights and biases
    rng(0);  % For reproducibility
    W_enc = 0.01 * randn(hidden_dim, input_dim);  % 10 x 98
    b_enc = zeros(hidden_dim, 1);                  % 10 x 1
    
    % Initialize decoder weights and biases
    W_dec = 0.01 * randn(input_dim, hidden_dim);   % 98 x 10
    b_dec = zeros(input_dim, 1);                   % 98 x 1
    
    N = size(X_all,1);  % number of samples
    
    for iter = 1:num_iter
        % Forward pass: Encoder
        Z = W_enc * X_all' + repmat(b_enc, 1, N);  % (10 x N)
        H = tanh(Z);  % latent representation (10 x N)
        
        % Forward pass: Decoder
        X_recon = tanh(W_dec * H + repmat(b_dec, 1, N));  % reconstructed input (98 x N)
        
        % Compute mean squared error loss
        diff = X_all' - X_recon;  % (98 x N)
        loss = sum(diff(:).^2) / N;
        
        % Backpropagation: Decoder
        dX_recon = -2 * diff / N;  % (98 x N)
        dZ_dec = dX_recon .* (1 - X_recon.^2);  % (98 x N)
        grad_W_dec = dZ_dec * H';  % (98 x 10)
        grad_b_dec = sum(dZ_dec, 2);  % (98 x 1)
        
        % Backpropagation: Encoder
        dH = W_dec' * dZ_dec;  % (10 x N)
        dZ_enc = dH .* (1 - H.^2);  % (10 x N)
        grad_W_enc = dZ_enc * X_all;  % (10 x 98)
        grad_b_enc = sum(dZ_enc, 2);  % (10 x 1)
        
        % Update weights and biases using gradient descent
        W_enc = W_enc - learning_rate * grad_W_enc;
        b_enc = b_enc - learning_rate * grad_b_enc;
        W_dec = W_dec - learning_rate * grad_W_dec;
        b_dec = b_dec - learning_rate * grad_b_dec;
        
        if mod(iter,200) == 0
            fprintf('Autoencoder iter %d, loss = %.4f\n', iter, loss);
        end
    end
    
    % Store autoencoder parameters
    modelParameters.W_enc = W_enc;
    modelParameters.b_enc = b_enc;
    modelParameters.W_dec = W_dec;
    modelParameters.b_dec = b_dec;
    
    %% 2c. Train direction-specific ridge regression models on the latent features.
    % Compute latent representations for all samples using the trained encoder.
    H_all = tanh(W_enc * X_all' + repmat(b_enc, 1, N))';  % (N x hidden_dim)
    
    lambda = 0.1;  % regularization parameter for ridge regression
    W_reg = cell(1, numDir);
    for dir = 1:numDir
        idx = (dir_labels == dir);
        H_dir = H_all(idx, :);  % latent features for this direction
        Y_dir = Y_all(idx, :);  % corresponding velocities [vel_x, vel_y]
        % Ridge regression closed-form: w = (H'H + lambda I) \ (H'Y)
        W_reg{dir} = (H_dir' * H_dir + lambda * eye(hidden_dim)) \ (H_dir' * Y_dir);
    end
    modelParameters.W_reg = W_reg;
end
