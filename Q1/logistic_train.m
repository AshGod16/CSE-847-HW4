function [weights] = logistic_train(data, labels, epsilon, maxiter)
%
% code to train a logistic regression classifier
%
% INPUTS:
% data = n * (d+1) matrix withn samples and d features, where
% column d+1 is all ones (corresponding to the intercept term)

% labels = n * 1 vector of class labels (taking values 0 or 1)

% epsilon = optional argument specifying the convergence
% criterion - if the change in the absolute difference in
% predictions, from one iteration to the next, averaged across
% input features, is less than epsilon, then halt
% (if unspecified, use a default value of 1e-5)

% maxiter = optional argument that specifies the maximum number of
% iterations to execute (useful when debugging in case your
% code is not converging correctly!)
% (if unspecified can be set to 1000)
%
% OUTPUT:
% weights = (d+1) * 1 vector of weights where the weights correspond to
% the columns of "data"
%


% load data
X = load("data.txt");
%display(class(X));
y = load("labels.txt");
%display(size(y));

X = [X ones(size(X,1), 1)];

% Split data into training and testing
% split_data.m contains the function split_data

% In the last parameter, enter the number of samples out of 4601 to be 
% included in training.
[x_train, x_test, y_train, y_test] = split_data(X, y, 2000);


% Define a constant learning rate
eta=0.05;


% zero-initialize the weights
w = zeros((size(x_train,2)), 1);

for i = 1:1000

% compute gradient
   
sig_inp = x_train * w;  % Input to the sigmoid function


% The following formulas used from the Logistic regression lecture notes.
b = 1 ./ (1 + exp(-sig_inp)) - y_train;  % Compute the element-wise sigmoid.
grad = (1/size(x_train,1)) * x_train.' * b;  % Gradient value.


% update weight
new_weight = w + eta * (-grad);
w = new_weight;


% check epsilon
pred = sign(x_train * w);  % get the new predictions
pred(pred==-1) = 0;  % Consider all -1s to be 0s

diff = sum(abs(pred - sign(sig_inp)))/size(x_train,1);  % Compute the \
                            % Difference between the predictions and the \
                            % original labels.

if abs(diff) < 1e-5  % If difference is less than epsilon, stop learning.  
   break
end
end

weights = w;   % Final weights


% Test on the test set
test_pred = sign(x_test * weights);  % Compute the test predcictions
test_pred(test_pred==-1) = 0;  % Consider all -1s as 0s


count = 0; % Count the number of correctly classified samples from the test set
for i = 1:size(test_pred, 1)
    if test_pred(i) == y_test(i)
        count = count + 1;
    end
end

display(count/size(y_test, 1) * 100);  % Display the accuracy percentage

