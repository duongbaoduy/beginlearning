function [f,g] = softmax_regression(theta, X,y)
  %
  % Arguments:
  %   theta - A vector containing the parameter values to optimize.
  %       In minFunc, theta is reshaped to a long vector.  So we need to
  %       resize it to an n-by-(num_classes-1) matrix.
  %       Recall that we assume theta(:,num_classes) = 0.
  %
  %   X - The examples stored in a matrix.  
  %       X(i,j) is the i'th coordinate of the j'th example.
  %   y - The label for each example.  y(j) is the j'th example's label.
  %
  m=size(X,2);
  n=size(X,1);

  % theta is a vector;  need to reshape to n x num_classes.
  theta=reshape(theta, n, []);
  num_classes=size(theta,2)+1;
  
  % initialize objective value and gradient.
  f = 0;
  g = zeros(size(theta));

  %
  % TODO:  Compute the softmax objective function and gradient using vectorized code.
  %        Store the objective function value in 'f', and the gradient in 'g'.
  %        Before returning g, make sure you form it back into a vector with g=g(:);
  %
%%% YOUR CODE HERE %%%
  %  首先将y转换为0,1值, 重新调整Theta
  yy = zeros(m, num_classes);
  I=sub2ind(size(yy), 1:size(yy,1), y);
  yy(I) = 1;
  theta(:, num_classes) = 0;
  g(:, num_classes) = 0;

  hvalue = exp(X'*theta);
  allSum = 1./sum(hvalue');
  allSum = repmat(allSum, 10, 1);
  hvalue = hvalue .* allSum';

  f = -1 * sum( sum(log(hvalue).*yy));
  g = -1 * X * (yy - hvalue);
  
  g = g(:, 1:num_classes-1);
%%% END OF YOUR CODE %%%
  g=g(:); % make gradient a vector for minFunc

