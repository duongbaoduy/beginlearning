imageDim = 64;         % image dimension
imageChannels = 3;     % number of channels (rgb, so 3)
patchDim = 8;          % patch dimension
visibleSize = patchDim * patchDim * imageChannels;  % number of input units 
outputSize = visibleSize;   % number of output units
hiddenSize = 400;           % number of hidden units 
epsilon = 0.1;	       % epsilon for ZCA whitening
poolDim = 19;          % dimension of pooling region
softmaxLambda = 1e-4;

load trainImages.mat;
numTrainImages = size(trainImages, 4);

load testImages.mat;
numTestImages = size(testImages, 4);


