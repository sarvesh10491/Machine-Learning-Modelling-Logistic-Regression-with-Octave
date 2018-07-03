function g = sigmoid(z)

%	SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

g = zeros(size(z));

g=1./(ones(size(z))+exp(-1.*z));

end
