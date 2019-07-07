#ifndef COMMON_H
#define COMMON_H

#include <cstdint>
//#include <limits>

const float fx = 517.3f;
const float fy =  516.5f;
const float cx =  318.6f;
const float cy =  255.3f;

const float distThres = 0.08f;
const float normalThres = 1.0f;
const float idealError = 0.0f;
const float intrinsics[] = {fx, 0, cx, 0, fy, cy, 0, 0, 1};
//const float intrinsics[] = {fx, 0, 0, 0, fy, 0, cx, cy, 1};
const int numCols =  640;
const int numRows =  480;
const int imgWidth = numCols;
const int imgHeight = numRows;


//const float MAXF = std::numeric_limits<float>::max();
//const float MINF = std::numeric_limits<float>::quiet_NaN();
//const float MINF = std::numeric_limits<float>::min();
const int pyramid_size = 2;
const int pyramid_iters[pyramid_size+1] = {4,5,10};

//#define MINF 0xff800000
//#define MAXF 0x7F7FFFFF

/*-----Kinect related------*/
const int kinZNear = 1.0f;
const float kinZFar = 1000.0f;

/*-----Hashtable Params----*/
const int numBuckets = 50000;
const int bucketSize = 5;
const int attachedLinkedListSize = 4;
const int numVoxelBlocks = 10000;
const int voxelBlockSize = 8;
const float voxelSize = 0.05f;
const int numOccupiedBlocks = 0;
const float maxIntegrationDistance = 4.0f;
const float truncScale = 0.01f;
const float truncation = 0.02f;
const int integrationWeightSample = 10;
const int integrationWeightMax = 255;


#endif
