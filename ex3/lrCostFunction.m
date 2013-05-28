function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples

pseudo_theta=theta;
pseudo_theta(1)=0;

p = sigmoid(X*theta);

J = (- log(p)' * y - log(1-p)' * (1-y)  + lambda/2*(pseudo_theta' * pseudo_theta))/m;

grad = (X' * (p - y) + lambda * pseudo_theta)/m;

end
