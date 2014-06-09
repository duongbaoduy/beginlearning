#ifndef _DETECTOR_H_
#define _DETECTOR_H_

#include "helper.h"

int DetectorInit();

int DetectorUpdateForResult(JNIEnv* env,
        const unsigned char* frameIn,
        jobject result,
        unsigned int wid, unsigned int hei );

#endif
