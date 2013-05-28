function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

p = sigmoid(X*theta);

J = (- log(p)' * y - log(1-p)'*(1-y))/m;

grad = 1/m * X' * (p - y);

end
