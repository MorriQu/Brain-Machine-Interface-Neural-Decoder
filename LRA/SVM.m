function svm_model = SVM(X, Y, kernel, C, tol, max_iteration)
    Y(Y==0) = -1; 
    a = zeros(size(X,1),1);
    E = zeros(size(X,1),1);
    b = 0;
    iter = 0;
    n = size(X,1);
    
    if contains(func2str(kernel), 'linearKernel')
        K = X * X';
    elseif strcmp(func2str(kernel), 'rbfKernel') 
        X2 = sum(X.^2, 2);
        K = X2 + X2' - 2 * (X * X');
        K = kernel(1, 0) .^ K;
    end
   
    while iter < max_iteration
        change_a = 0;
        for i = 1:n
            E(i) = b + sum(a .* Y .* K(:,i)) - Y(i);
            if ((Y(i)*E(i) < -tol && a(i) < C) || (Y(i)*E(i) > tol && a(i) > 0))
                j = ceil(n * rand());
                while j == i
                    j = ceil(n * rand());
                end
                E(j) = b + sum(a .* Y .* K(:,j)) - Y(j);
                old_a_i = a(i);
                old_a_j = a(j);
                if Y(i) == Y(j)
                    L = max(0, a(j) + a(i) - C);
                    H = min(C, a(j) + a(i));
                else
                    L = max(0, a(j) - a(i));
                    H = min(C, C + a(j) - a(i));
                end
                a(j) = a(j) - (Y(j)*(E(i)-E(j)))/(2*K(i,j)-K(i,i)-K(j,j));
                a(j) = min(max(a(j), L), H);
                if abs(a(j)-old_a_j) < tol
                    a(j) = old_a_j;
                    continue;
                end
                a(i) = a(i) + Y(i)*Y(j)*(old_a_j - a(j));
                b1 = b - E(i) - Y(i)*(a(i)-old_a_i)*K(i,i) - Y(j)*(a(j)-old_a_j)*K(i,j);
                b2 = b - E(j) - Y(i)*(a(i)-old_a_i)*K(i,j) - Y(j)*(a(j)-old_a_j)*K(j,j);
                if (a(i) > 0 && a(i) < C)
                    b = b1;
                elseif (a(j) > 0 && a(j) < C)
                    b = b2;
                else
                    b = (b1+b2)/2;
                end
                change_a = change_a + 1;
            end
        end
        if change_a == 0
            iter = iter + 1;
        else
            iter = 0;
        end
    end
    idx = a > 0;
    svm_model.y = Y(idx);
    svm_model.X = X(idx,:);
    svm_model.b = b;
    svm_model.a = a(idx);
    svm_model.w = ((a.*Y)'*X)';
    svm_model.kernel = kernel;
end