function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.01;
sigma = 0.03;

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
##init = false;
##bestAvg = 0;
##bestC = 1;
##bestSigma = 1;
##values = [0.01 0.03 0.1 0.3 1 3 10 30];
##for cP = 1:columns(values)
##  for sP = 1:columns(values)
##    newC = values(1, cP);
##    newSigma = values(1, sP);
##    model= svmTrain(X, y, newC, @(x1, x2) gaussianKernel(x1, x2, newSigma));
##    predictions = svmPredict(model, Xval);
##    avg = mean(double(predictions ~= yval));
##    ##printf(["C = %f sigma = %f mean = %f"], newC, newSigma, avg);
##    if(avg < bestAvg || ~init)
##      init = true;
##      bestAvg = avg;
##      bestC = newC;
##      bestSigma = newSigma;
##    endif
##  endfor
##endfor
##C = bestC;
##sigma = bestSigma;
##printf(["best C = %f sigma = %f \n"], C, sigma);
% =========================================================================
C = 1;
sigma = 0.1;
end
