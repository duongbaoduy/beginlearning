function [ cost, grad, pred_prob] = supervised_dnn_cost( theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
hAct = cell(numHidden+2, 1);
gradStack = cell(numHidden+1, 1);
%% forward prop
%%% YOUR CODE HERE %%%
a = data;
hAct{1} = data;
for i=1:numHidden+1
    a(size(a,1)+1,:) = 1;
    w = [stack{i}.W stack{i}.b];

    z = w * a;
    if ( i < (numHidden + 1) )
        a = 1 ./ ( 1 + exp(-1*z) ); 
    else
        a = exp(z);
    end
    hAct{i+1} = a;
end

m = size( hAct{numHidden+2}, 2);
num_classes = size(hAct{numHidden+2}, 1);
hvalue = hAct{numHidden + 2};
allSum = 1./sum(hvalue);
allSum = repmat(allSum, num_classes, 1);
hvalue = hvalue .* allSum;
pred_prob = hvalue; 

%% return here if only predictions desired.
if po
  cost = -1; ceCost = -1; wCost = -1; numCorrect = -1;
  grad = [];  
  return;
end;

%% compute cost
%%% YOUR CODE HERE %%%
y = zeros(m, num_classes);
I = sub2ind(size(y), 1:size(y,1), labels');
y(I) = 1;
cost = -1 * sum ( sum ( log(hvalue) .* y'));

%% compute gradients using backpropagation
%%% YOUR CODE HERE %%%
delta_next = -1 * ( y - hvalue');
gradStack{numHidden+1} = struct;
gradStack{numHidden+1}.W = (hAct{numHidden+1} * delta_next)' / m;
gradStack{numHidden+1}.b = (sum(delta_next)/m)';

for i=numHidden+1:-1:2
    a = hAct{i};
    delta = stack{i}.W' * delta_next';
    delta = delta .* a .* ( 1-a);
    delta_next = delta';
    
    gradStack{i-1} = struct; 
    gradStack{i-1}.W = (hAct{i-1} * delta_next)'/m; 
    gradStack{i-1}.b = (sum(delta_next)/m)';
end

%% compute weight penalty cost and gradient for non-bias terms
%%% YOUR CODE HERE %%%

%% reshape gradients into vector
[grad] = stack2params(gradStack);
end



