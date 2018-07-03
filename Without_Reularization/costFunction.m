function [J, grad] = costFunction(theta, X, y)

%   Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));


h=sigmoid(X*theta);		% Find sigmoid


% Find cost
cost = ((-1.*y)'*(log(h))) - ((1-y)'*(log(1-h)));
J = (1/m)*(sum(cost));

% Find gradient
t=X'*(h-y);
grad = ((1/m)*(t));

end
