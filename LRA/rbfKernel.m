function kernel = rbfKernel(X1, X2)
% RBF (Gaussian) Kernel function for SVM

sigma = 0.05;
kernel = exp(-1 * (X1-X2)'* (X1-X2) / (2 * sigma * sigma));

end
