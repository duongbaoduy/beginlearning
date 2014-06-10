#include "detector.h"

extern double wData[];
extern double bData[];
extern double meanData[];
extern double whiteData[];
extern double optThetaData[];


const int DogDetector::inputImageSize = 64;
const int DogDetector::convPatchSize = 8;
const int DogDetector::visualSize = 192;
const int DogDetector::hiddenSize = 400;
const int DogDetector::poolSize = 19;
const int DogDetector::featureSize = 3600 + 1;

DogDetector::DogDetector() {
    loadFeatureMatrix();    
}
DogDetector::~DogDetector() {
}

double DogDetector::detect(std::vector<Eigen::MatrixXd>& sourcePatches) {
    // Convolved input raw images to 400x56x56 size
    std::vector<Eigen::MatrixXd> features;
    for(int i = 0; i < hiddenSize; i++) {
        for (int y = 0; y <= inputImageSize - convPatchSize; y++) {
            for (int x = 0; x <= inputImageSize - convPatchSize; x++) {
                Eigen::MatrixXd patch(convPatchSize, convPatchSize * 3);                
                Eigen::MatrixXd patch_r = sourcePatches[0].block(y,x,convPatchSize, convPatchSize);
                Eigen::MatrixXd patch_g = sourcePatches[1].block(y,x,convPatchSize, convPatchSize);
                Eigen::MatrixXd patch_b = sourcePatches[2].block(y,x,convPatchSize, convPatchSize);
                patch << patch_r , patch_g, patch_b;
            }
        } 
    } 

    return 0.5;
}

void DogDetector::loadFeatureMatrix() {
    featureW = new Eigen::MatrixXd(hiddenSize, visualSize);
    for(int i = 0; i < visualSize; i++) {
        for(int j = 0; j < hiddenSize; j++) {
            (*featureW)(j,i) = wData[i*hiddenSize + j];
        }
    }
    
    featureB = new Eigen::MatrixXd(hiddenSize, 1);
    for(int i = 0; i < hiddenSize; i++) {
        (*featureB)(i,0) = bData[i];
    }

    zcaMean = new Eigen::MatrixXd(visualSize, 1);
    for(int i = 0; i < visualSize; i++) {
        (*zcaMean)(i,0) = meanData[i];
    }

    zcaWhite = new Eigen::MatrixXd(visualSize, visualSize); 
    for(int i = 0; i < visualSize; i++) {
        for (int j = 0; j < visualSize; j++) {
            (*zcaWhite)(j,i) = whiteData[i*visualSize+j];
        }
    }

    optTheta = new Eigen::MatrixXd(featureSize, 1);
    for(int i = 0; i < featureSize; i++) {
        (*optTheta)(i,0) = optThetaData[i];
    }
}

