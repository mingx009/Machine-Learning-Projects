function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
dh2 = (X*theta-y)'*(X*theta-y) ; 
theta2 = theta(2:end)'*theta(2:end) ;
J = (dh2 + lambda*theta2)/(2*m) ;


gtemp = (X*theta-y)'*X   ;
gtemp(2:end) = gtemp(2:end) + lambda*theta(2:end)'   ;

grad = gtemp'/m  ;












% =========================================================================

grad = grad(:);

end
