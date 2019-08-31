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

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
n =  size(Theta1,2);
p =  size(Theta2,2);        
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
Delta1=zeros(size(Theta1));
Delta2=zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


X=[ones(m,1) X];
act=X*(Theta1)';
act1=[ones(m,1) act];

for i=1:(size(act,2)),
act(:,i)=sigmoid(act(:,i));
end

act=[ones(m,1) act];

out=(act*(Theta2)');

for i=1:(size(out,2)),

out(:,i)=sigmoid(out(:,i));
 
end

out1=log(out);
out2=log(1-out);


 I = eye(num_labels);
Y = zeros(m, num_labels);


for i=1:m
  Y(i, :)= I(y(i), :);
end


  J = (1/m)*sum(sum((-Y).*(out1) - (1-Y).*(out2), 2));
  
  J = (1/m)*sum(sum((-Y).*(out1) - (1-Y).*(out2), 2))+(((sum(sum((Theta1(:,(2:n)).^2),2)))+sum(sum((Theta2(:,(2:p)).^2),2)))*(lambda/(2*m)));
  
  
  for i=1:m,
  
    y_c=zeros(num_labels,1);
	 y_c(y(i))=y(i);
     y_c=((y_c)==y(i));	 
  	
	
	 delta3=(out(i,:))' - (y_c);    %(10*1)
	 

	 

	 delta2=(((Theta2)' * delta3)) .* (sigmoidGradient(act1(i,:)'));
	 
	 
	 delta2=delta2(2:length(delta2));   %(25*1) 
	 

	 
	 Delta1=Delta1+((delta2)*(X(i,:)));
	 Delta2=Delta2+((delta3)*(act(i,:)));

	
 end; 
 
 



theta1=Theta1;
theta2=Theta2;
theta1(:,1)=zeros(size(Theta1,1),1);
theta2(:,1)=zeros(size(Theta2,1),1);

Theta1_grad =(Delta1)/(m)+((lambda/m)*theta1); 
Theta2_grad =(Delta2)/(m)+((lambda/m)*theta2); 




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
