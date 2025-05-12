function preds = SVMPred(model_param, X)
    nSV = size(model_param.X,1);
    nTest = size(X,1);
    preds = zeros(nTest,1);
    for i = 1:nTest
        K = zeros(nSV,1);
        for j = 1:nSV
            K(j) = model_param.kernel(X(i,:), model_param.X(j,:));
        end
        f = sum(model_param.a .* model_param.y .* K) + model_param.b;
        preds(i) = (f >= 0);  
    end
end
