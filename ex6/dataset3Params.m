function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
%C = 1;
%sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
C_test = [0.01,0.03,0.1,0.3,1,3,10,30];
sigma_test = [0.01,0.03,0.1,0.3,1,3,10,30];
err = zeros(size(C_test,2)*size(sigma_test,2),1);
i = 1;
for C = C_test
    for sigma = sigma_test
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        yval_h = svmPredict(model,Xval);
        flag = yval_h == yval;
        err(i,1) = length(find(flag == 0));
        i = i + 1;
    end
end
index = find(err == min(err));
C = C_test(1,ceil(index/size(sigma_test,2)));
sigma = sigma_test(1,mod(index,size(C_test,2)));






% =========================================================================

end
