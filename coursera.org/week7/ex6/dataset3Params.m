function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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

possible_values = [0.01 0.03 0.1 0.3 1 3 10 30];

x1 = [1 2 1]; x2 = [0 4 -1];

trainingSize = size(possible_values, 2);
err = zeros(trainingSize, trainingSize);

for i=1:trainingSize,
    for j=1:trainingSize,
        model = svmTrain(X, y, possible_values(i), @(x1, x2) gaussianKernel(x1, x2, possible_values(j)));
        predictions = svmPredict(model, Xval);
        err(i, j) = mean(double(predictions ~= yval));
    end;
end;

[min_err, row] = min(min(err, [], 2));
[min_err, column] = min(min(err, [], 1));

C = possible_values(row);
sigma = possible_values(column);


% =========================================================================

end
