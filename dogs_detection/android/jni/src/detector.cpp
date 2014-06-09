#include <vector>
#include <android/bitmap.h>
#include <bv/image.h>
#include <bv/image_convert.h>
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

    double detect(std::vector<Eigen::MatrixXd>& sourcePatches) {
        return 0.5;
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

public:
    static const int inputImageSize;
    static const int convPatchSize;
    static const int visualSize;
    static const int hiddenSize;
    static const int poolSize;
    static const int featureSize;
};

const int DogDetector::inputImageSize = 64;
const int DogDetector::convPatchSize = 8;
const int DogDetector::visualSize = 192;
const int DogDetector::hiddenSize = 400;
const int DogDetector::poolSize = 19;
const int DogDetector::featureSize = 3600 + 1;


DogDetector* myDetector = NULL;
/********************************************************************************/
#define SATURATE(a,min,max) ((a)<(min)?(min):((a)>(max)?(max):(a)))
static void yuvToRgb(unsigned char inY, unsigned char inU, unsigned char inV,
        unsigned char& R, unsigned char& G, unsigned char& B) {
    int Gi = 0, Bi = 0;
    int Y = 9535*(inY-16);
    int U = inU - 128;
    int V = inV - 128;
    int Ri = (Y + 13074*V) >> 13;
    Ri = SATURATE(Ri, 0, 255);
    Gi = (Y - 6660*V - 3203*U) >> 13;
    Gi = SATURATE(Gi, 0, 255);
    Bi = (Y + 16531*U) >> 13;
    Bi = SATURATE(Bi, 0, 255);
    R = Ri;
    G = Gi;
    B = Bi;
}
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

    int rectangleSize = hei/2;
    // 将YUV图像转换为64x64的RGB矩阵 
    std::vector<Eigen::MatrixXd> sourcePatches;
    sourcePatches.resize(3);
    for(int i = 0; i < 3; i++) {
        sourcePatches[i].resize(rectangleSize, rectangleSize);
    }
    int beginX = wid/2 - rectangleSize/2;
    int beginY = hei/2 - rectangleSize/2;
    for (int y = beginY; y < beginY + rectangleSize; y++) {
        for (int x = beginX; x < beginX + rectangleSize; x++)   {
            unsigned char Y = frameIn[y * wid + x];
            unsigned char V = frameIn[y/2 * wid + (x&0xFFFFFFFE) + wid*hei];
            unsigned char U = frameIn[y/2 * wid + (x&0xFFFFFFFE) + 1 + wid*hei];
            unsigned char r,g,b;
            yuvToRgb(Y,U,V,r,g,b);
            sourcePatches[0](x-beginX,y-beginY) = r/255.0;
            sourcePatches[1](x-beginX,y-beginY) = g/255.0;
            sourcePatches[2](x-beginX,y-beginY) = b/255.0;
        }
    }
    // resize to 64x64
    Eigen::MatrixXd targetPatch(DogDetector::inputImageSize, DogDetector::inputImageSize);
    for(int i = 0; i < 3; i++) {
        bv::Convert::resizeImage(sourcePatches[i], targetPatch);
        sourcePatches[i] = targetPatch.transpose();
    }
    
    double likeDog = myDetector->detect(sourcePatches); 
    LOGD(">>>>>>>>>>>>>>>>> %f", likeDog);

    // 修改输出图像 
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

