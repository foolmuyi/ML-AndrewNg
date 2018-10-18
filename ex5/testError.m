clc;
lambda = 3;
theta = trainLinearReg(X_poly,y,lambda);
h_test = X_poly_test * theta;
t = size(X_poly_test,1);
error_test = ((h_test - ytest)'*(h_test - ytest))/2/t;
error_test