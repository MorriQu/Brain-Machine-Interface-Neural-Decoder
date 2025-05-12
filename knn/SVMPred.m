function preds = SVMPred(model_param, X)
    % Number of support vectors and test samples
    nSV   = size(model_param.X, 1);
    nTest = size(X,           1);

    % Initialize decision values and binary predictions
    pred  = zeros(nTest,1);
    preds = zeros(nTest,1);

    kernelName = func2str(model_param.kernel);

    if strcmp(kernelName, 'linearKernel')
        pred = X * model_param.w + model_param.b;

    elseif strcmp(kernelName, 'rbfKernel')
        X_1 = sum(X .^ 2, 'all');   
        X_2 = sum(model_param.X .^ 2, 2)';   
        K = X_1 + X_2 - 2 * (X * model_param.X');
        gamma = model_param.kernel(1, 0);
        K     = gamma .^ K;
        K     = model_param.y' .* K;
        K     = model_param.a' .* K;
        pred  = sum(K, 2);
    end

    preds(pred >  0) = 1;
    preds(pred <= 0) = 0;
end
