function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%
%   fprintf('\nFeedforward Using Neural Network ...\n')
% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
%fprintf('\nFeedforward Using Neural Network ...\n')
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
X = [ones(m,1) X];
  
    z2 = X*Theta1'   ;    %% row-vector matrix
    a2 = sigmoid(z2)  ;

    a2 = [ones(m, 1) a2]  ;  %% add bias layer
    
    z3 = a2*Theta2'  ;      %% row-vector matrix
    a3 = sigmoid(z3);       %% output row-vector matrix

YY = zeros(m,num_labels);


for ii=1:m
    YY(ii,y(ii))=1;        %% label out
end

for ii=1:m
    J = J + log(a3(ii,:))*YY(ii,:)'  + log(1-a3(ii,:))*(1-YY(ii,:))'  ;
end
    J = -J;
    
%% Regularization
%% layer1
reg1 = 0;
 for jj =1:hidden_layer_size
     for kk=2:input_layer_size+1
         reg1 = reg1 + Theta1(jj,kk)^2;
     end
 end
 
 reg2 = 0;
 
 for jj = 1:num_labels
     for kk=2:hidden_layer_size+1
         reg2 = reg2 + Theta2(jj,kk)^2;
     end
 end
 
 J = (J + (reg1+reg2)*lambda/2)/m;
 
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
g_sum1 = 0.0 ;
g_sum2 = 0.0 ;
z22 = [ones(m,1) z2];

for ii=1:m
    
 Delta3 = a3(ii,:) - YY(ii,:)   ;
 Delta3 = Delta3'   ;    %% column vector
 Delta2 = Theta2'*Delta3.*sigmoidGradient(z22(ii,:)')    ; %% column vector  
 % Theta2'*Delta3' --matrix(# of activation units in layer2 * # of sample
 % m)
 Delta2 =  Delta2(2:end)    ;   
 
 g_sum2 = g_sum2 + Delta3 * a2(ii,:);
 g_sum1 = g_sum1 + Delta2 * X(ii,:) ; 
 
end

g_sum2 = g_sum2/m  ;
g_sum1 = g_sum1/m  ;
 
Theta1_grad = g_sum1 ;
Theta2_grad = g_sum2 ;
    


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%% For layer1
for j=2:input_layer_size + 1
    Theta1_grad(:,j) = Theta1_grad(:,j) + lambda/m*Theta1(:,j);
end

for j=2:hidden_layer_size + 1
    Theta2_grad(:,j) = Theta2_grad(:,j) + lambda/m*Theta2(:,j);
end
    














% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
