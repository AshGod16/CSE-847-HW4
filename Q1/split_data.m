function [x_train, x_test, y_train, y_test] = split_data(X, labels, n)

%display(size(X(1, :)));
x_train=X(1:n, :);
y_train=labels(1:n, :);
x_test=X(n+1:size(X,1), :);
y_test=labels(n+1:size(X,1), :);

end