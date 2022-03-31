function [w, c] = logistic_l1_train(data, labels, par)
% OUTPUT w is equivalent to the first d dimension of weights in logistic train
% c is the bias term, equivalent to the last dimension in weights in logistic train.
% Specify the options (use without modification).

X_train = struct2array(load("ad_data.mat", "X_train"));
y_train = struct2array(load("ad_data.mat", "y_train"));
X_test = struct2array(load("ad_data.mat", "X_test"));
y_test = struct2array(load("ad_data.mat", "y_test"));

display(size(y_train));

par  = [1e-8, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];

opts.rFlag = 1; % range of par within [0, 1].
opts.tol = 1e-6; % optimization precision
opts.tFlag = 4; % termination options.
opts.maxIter = 5000; % maximum iterations

for i = 1:length(par)

    [w, c] = LogisticR(X_train, y_train, par(i), opts);

    
    preds = sign(X_test * w + c);

    
    % Count the number of features used.
    count = 0;
    for j = 1:length(w)
        if w(j) ~= 0
            count = count + 1;
        end
    end


    [x, y, t, auc] = perfcurve(y_test, preds, -1);
    display(count);
    display(auc);


end
end