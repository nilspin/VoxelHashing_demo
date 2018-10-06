#ifndef CAMERA_TRACKING_H
#define CAMERA_TRACKING_H

#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_CUDA
#define GLM_SWIZZLE

#include <Windows.h>
#include <cuda_runtime_api.h>
//This is a simple vector library. Use this with CUDA instead of GLM.
#include "cuda_helper/cuda_SimpleMatrixUtil.h"
#include "LinearSystem.h"
#include "EigenUtil.h"

class CameraTracking  {

private:
  int width, height;
  LinearSystem linearSystem;
  Eigen::Matrix4f temp;;
  float4* d_correspondenceNormals;
  float4* d_correspondence;
  float4x4 deltaTransform;
public:
  
  CameraTracking(int, int);
  ~CameraTracking();
  void Align(float4*, float4*, float4*, float4*, const uint16_t*, const uint16_t*);
  Eigen::Matrix4f rigidAlignment(const float4*, const float4*, const Eigen::Matrix4f&);
};

#endif 