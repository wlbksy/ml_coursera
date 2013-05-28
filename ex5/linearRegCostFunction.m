function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

pseudo_theta = theta;
pseudo_theta(1) = 0;

bias = X*theta-y ;

J = (bias' * bias + lambda * (pseudo_theta' * pseudo_theta))/2/m;

grad = (X' * bias + lambda * pseudo_theta)/m;











% =========================================================================

grad = grad(:);

end
