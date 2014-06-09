#include <android/bitmap.h>
#include <bv/image.h>
#include "helper.h"

extern double wData[];
extern double bData[];
extern double meanData[];
extern double whiteData[];
extern double optThetaData[];

class DogDetector {
public:
    DogDetector() {
        loadFeatureMatrix();    
    }
    ~DogDetector() {
    
    }

private:
    void loadFeatureMatrix() {
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
private:
    Eigen::MatrixXd* featureW;
    Eigen::MatrixXd* featureB;
    Eigen::MatrixXd* zcaMean;
    Eigen::MatrixXd* zcaWhite; 
    
    Eigen::MatrixXd* optTheta;

private:
    static const int inputImageSize = 64;
    static const int inputImageChannel = 3;
    static const int convPatchSize = 8;
    static const int visualSize = convPatchSize * convPatchSize * inputImageChannel;
    static const int hiddenSize = 400;
    static const int poolSize = 19;
    static const int featureSize = 9 * hiddenSize + 1;
};

DogDetector* myDetector = NULL;

/********************************************************************************/
int DetectorInit() {
    if ( myDetector != NULL ) {
        delete myDetector;
        myDetector = NULL;
    }
    myDetector = new DogDetector();


    return 0;
}

int DetectorUpdateForResult(JNIEnv* env,
        const unsigned char* frameIn,
        jobject bitmap,
        unsigned int wid, unsigned int hei ) {
    
    int ret; 
    AndroidBitmapInfo  info;
    unsigned int*              pixels;
    if ((ret = AndroidBitmap_getInfo(env, bitmap, &info)) < 0) {
        LOGD("AndroidBitmap_getInfo() failed ! error=%d", ret);
        return -1;
    }
    
    if ((ret = AndroidBitmap_lockPixels(env, bitmap, (void**)&pixels)) < 0) {
        LOGD("AndroidBitmap_lockPixels() failed ! error=%d", ret);
        return -1;
    } 
    
    int rectangleSize = 256;
      
    int lineStride = info.stride / 4;
    for(unsigned int x = (wid - rectangleSize)/2; x < (wid + rectangleSize)/2; x++) {
        int y = (hei - rectangleSize)/2;
        pixels[(y-1)*lineStride+x] = 0xFFFFFFFF;
        pixels[y*lineStride+x] = 0xFFFFFFFF;
        pixels[(y+1)*lineStride+x] = 0xFFFFFFFF;
        y = (hei + rectangleSize) / 2;
        pixels[(y-1)*lineStride+x] = 0xFFFFFFFF;
        pixels[y*lineStride+x] = 0xFFFFFFFF;
        pixels[(y+1)*lineStride+x] = 0xFFFFFFFF;
    }
    for(unsigned int y = (hei - rectangleSize)/2; y < (hei + rectangleSize)/2; y++) {
        int x = (wid - rectangleSize)/2;
        pixels[y*lineStride+x-1] = 0xFFFFFFFF;
        pixels[y*lineStride+x] = 0xFFFFFFFF;
        pixels[y*lineStride+x+1] = 0xFFFFFFFF;
        x = (wid + rectangleSize) / 2;
        pixels[y*lineStride+x-1] = 0xFFFFFFFF;
        pixels[y*lineStride+x] = 0xFFFFFFFF;
        pixels[y*lineStride+x+1] = 0xFFFFFFFF;
    }

    AndroidBitmap_unlockPixels(env, bitmap);
    return 0;
}


