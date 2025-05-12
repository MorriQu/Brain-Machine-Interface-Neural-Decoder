function kernelVal = rbfKernel(X1, X2, sigma)
    diff = X1 - X2;
    kernelVal = exp(- (diff*diff')/(2*sigma^2));
end

