function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
for ii = 1:num_movies
    for jj = 1:num_users
        if R(ii,jj) == 1
            J = J + (Theta(jj,:)*X(ii,:)' - Y(ii,jj))^2   ;
        end
    end
end

% add regularization
% parameter vectors
P_v = 0.0 ;
for jj =1:num_users
    for kk =1: num_features
        P_v = P_v + Theta(jj,kk)^2 ;
    end
end
% feature vectors
F_v = 0.0 ;
for ii = 1 : num_movies
    for kk =1:num_features
        F_v = F_v + X(ii,kk)^2   ;
    end
end

J = J + (P_v + F_v)*lambda ;

J = J/2 ;




% X_grad
for ii = 1 : num_movies
    for kk = 1 : num_features
        for jj = 1:num_users
            if R(ii,jj) == 1
               X_grad(ii,kk) = X_grad(ii,kk) + (Theta(jj,:)*X(ii,:)' - Y(ii,jj))*Theta(jj,kk);
            end
        end
    end
end
X_grad = X_grad + lambda*X ;


% Theta_grad
for jj = 1 : num_users
    for kk = 1 : num_features
        for ii = 1 : num_movies
            if R(ii,jj) == 1
               Theta_grad(jj,kk) = Theta_grad(jj,kk) + (Theta(jj,:)*X(ii,:)' - Y(ii,jj))*X(ii,kk);
            end
        end
    end
end

Theta_grad = Theta_grad + lambda*Theta;



% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
