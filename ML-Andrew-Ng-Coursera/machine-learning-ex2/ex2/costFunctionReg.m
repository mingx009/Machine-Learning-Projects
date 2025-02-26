function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
hh = zeros(m);
hh = sigmoid(X*theta);
hh_theta = hh;
hh = hh - y;
grad = X'*hh;
grad = grad/m;
grad(2:size(theta),1) = grad(2:size(theta),1)+lambda/m*theta(2:size(theta),1);

% for cost function

J = y'*log(hh_theta)+(1-y)'*log(1-hh_theta);
J = -J/m +lambda/(2*m)*theta(2:size(theta),1)'*theta(2:size(theta),1) ;





% =============================================================

end
