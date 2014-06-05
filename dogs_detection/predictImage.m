function value = predictImage(fileName);

imageDim = 64;         % image dimension
imageChannels = 3;     % number of channels (rgb, so 3)

patchDim = 8;          % patch dimension
visibleSize = patchDim * patchDim * imageChannels;  % number of input units 
hiddenSize = 400;           % number of hidden units 
poolDim = 19;          % dimension of pooling region

load dogFeatures.mat;
W = reshape(optTheta(1:visibleSize * hiddenSize), hiddenSize, visibleSize);
b = optTheta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);

img = imread(fileName);
img = im2double(img);
[w h c] = size(img);
assert( w == 64 && h == 64, 'We only support 64x64 test image');
images = zeros(64,64,3,0);
images(:,:,:,1) = img;

numTestImages = size(images,4);

imshow(images(:,:,:,1));

convolvedFeatures = cnnConvolve(patchDim, hiddenSize, ...
        images, W, b, ZCAWhite, meanPatch);
pooledFeatures = cnnPool(poolDim, convolvedFeatures);

load linearRegOptTheta.mat
X = permute(pooledFeatures, [1 3 4 2]);
X = reshape(X, numel(pooledFeatures) / numTestImages,...
        numTestImages);
X = [ones(1, size(X,2)); X];

value = sigmoid(X' * optLinearRegTheta);

end
