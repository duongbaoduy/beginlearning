
lambda = 0.01;

load cnnPooledFeatures.mat;

numTrainImages = size(pooledFeaturesTrain, 2);
X = permute(pooledFeaturesTrain, [1 3 4 2]);
X = reshape(X, numel(pooledFeaturesTrain) / numTrainImages,...
        numTrainImages);
X = [ones(1, size(X,2)); X];
theta = 0.005 * randn(size(X, 1), 1);

addpath minFunc/
options = struct;
options.Method = 'lbfgs';
options.maxIter = 400;
minFuncOptions.display = 'on';

[optLinearRegTheta, cost] = minFunc( @(p) linearRegCost(p, ...
                                   X', trainLabels', lambda), ... 
                              theta, options);
Y = sigmoid(X' * optLinearRegTheta);
Y = (Y>0.5);
acc = (Y(:) == trainLabels(:));
acc = sum(acc) / size(acc, 1);
fprintf('Accuracy of train: %2.3f%%\n', acc * 100);

numTestImages = size(pooledFeaturesTest, 2);
X = permute(pooledFeaturesTest, [1 3 4 2]);
X = reshape(X, numel(pooledFeaturesTest) / numTestImages,...
        numTestImages);
X = [ones(1, size(X,2)); X];
testY = sigmoid(X' * optLinearRegTheta);
testY = (testY>0.5);
acc = sum(testY) / size(testY,1);
fprintf('Accuracy of test: %2.3f%%\n', acc * 100);

save('linearRegOptTheta.mat', 'optLinearRegTheta');
