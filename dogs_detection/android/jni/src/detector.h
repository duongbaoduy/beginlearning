#ifndef _DETECTOR_H_
#define _DETECTOR_H_

#include <vector>
#include <Eigen/Core>

class DogDetector {
public:
    DogDetector();
    ~DogDetector();

    double detect(std::vector<Eigen::MatrixXd>& sourcePatches);

private:
    void loadFeatureMatrix();

private:
    Eigen::MatrixXd* featureW;
    Eigen::MatrixXd* featureB;
    Eigen::MatrixXd* zcaMean;
    Eigen::MatrixXd* zcaWhite; 
    Eigen::MatrixXd* optTheta;

public:
    static const int inputImageSize;
    static const int convPatchSize;
    static const int visualSize;
    static const int hiddenSize;
    static const int poolSize;
    static const int featureSize;
};


#endif
