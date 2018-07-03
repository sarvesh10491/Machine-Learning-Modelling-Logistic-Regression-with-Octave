function [J, grad] = costFunctionReg(theta, X, y, lambda)

%	COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

% Calculate theta with regularization applied for values j>1
p=length(theta);
nt1=0;
nt2=(lambda/m)*(theta(2:p));
nt=[nt1;nt2];

h=sigmoid(X*theta);


% Calculate cost
cost = ((-1.*y)'*(log(h))) - ((1-y)'*(log(1-h)));
J_1 = (1/m)*(sum(cost));
k=(theta.*theta);
k=[nt1;k(2:p)];
J_2 = (lambda/(2*m))*(sum(k));
J = J_1+J_2;


% Calculate gradient
t=X'*(h-y);
grad = ((1/m)*(t)) + (nt);


end
