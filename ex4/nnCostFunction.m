function [J, grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

m = size(X, 1);

% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
X = [ones(m, 1) X];
z2 = X*Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);

diag_labels = eye(num_labels);
y_labels = diag_labels(y,:);

bias = -log(a3) .* y_labels - log(1-a3) .* (1-y_labels);

pseudo_theta1 = Theta1;
pseudo_theta2 = Theta2;
pseudo_theta1(:,1)=0;
pseudo_theta2(:,1)=0;
pseudo_theta = [pseudo_theta1(:) ; pseudo_theta2(:)];


J = (sum(sum(bias)) + lambda/2 * (pseudo_theta' * pseudo_theta))/m;


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.

% Part 3: Implement regularization with the cost function and gradients.

delta3 = a3 - y_labels;
z2=[ones(m,1) z2];
delta2 = delta3 * Theta2 .* sigmoidGradient(z2);
delta2 = delta2(:,2:end);

Theta1_grad = (delta2' * X + lambda * pseudo_theta1)/ m;
Theta2_grad = (delta3' * a2 + lambda * pseudo_theta2)/ m;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
