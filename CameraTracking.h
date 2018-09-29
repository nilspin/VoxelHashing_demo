#ifndef CAMERA_TRACKING_H
#define CAMERA_TRACKING_H

#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_CUDA
#define GLM_SWIZZLE

#include<Windows.h>
//#include<glm/glm.hpp>
#include<cuda_runtime_api.h>
#include "cuda_helper/helper_cuda.h"
#include "cuda_helper/helper_math.h"

//This is a simple vector library. Use this with CUDA instead of GLM.
#include "cuda_helper/cuda_SimpleMatrixUtil.h"


class CameraTracking  {

private:
  int width, height;
  
  float4* d_correspondenceNormals;
  float4* d_correspondence;
  float4x4 deltaTransform;
  void preProcess(float4 *, float4*, const uint16_t*);
public:
  
  CameraTracking(int, int);
  ~CameraTracking();
  //void FindCorrespondences(const float4*, const float4*, const float4*, const float4*, float4*, float4*, const float4x4&, int, int);
  void Align(float4*, float4*, float4*, float4*, const uint16_t*, const uint16_t*);
};

__global__
void FindCorrespondences(const float4*, const float4*, const float4*, const float4*, float4*, float4*, const float4x4, int, int);

__device__
static inline int2 cam2screenPos(const float3&);

__global__
void calculateVertexPositions(float4* , const uint16_t*);

__global__
void calculateNormals(const float4* , float4*);

#endif 