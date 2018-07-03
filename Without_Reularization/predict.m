function p = predict(theta, X)

%	PREDICT Predict whether the label is 0 or 1 using learned logistic 
%	regression parameters theta
%   p = PREDICT(theta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if sigmoid(theta'*x) >= 0.5, predict 1)

m = size(X, 1); % Number of training examples
p = zeros(m, 1);


p = X*theta;


% Use sigmoid function to classify
sp=sigmoid(p);

sp(sp<0.5) = 0;
sp(sp>=0.5) = 1;

p=sp;

end
