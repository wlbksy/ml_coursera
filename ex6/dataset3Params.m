function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.

range = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
cost = zeros(8,8);
for i = 1:8
    C = range(i);
    for j = 1:8
        sigma = range(j);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
        predictions = svmPredict(model, Xval);
        cost(i,j) = mean(double(predictions ~= yval));
    end
end

min_cost =  min(min(cost));
idx = find(cost == min_cost);

C = range(mod(idx,8));
sigma = range(ceil(idx/8));

end
