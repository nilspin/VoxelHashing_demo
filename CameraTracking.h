#ifndef CAMERA_TRACKING_H
#define CAMERA_TRACKING_H

#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_CUDA
#define GLM_SWIZZLE

#if defined(_WIN32)
  #include <Windows.h>
#endif

#include <cuda_runtime_api.h>
//#include <thrust/device_vector.h>
//#include <thrust/device_ptr.h>

//This is a simple vector library. Use this with CUDA instead of GLM.
#include "cuda_helper/cuda_SimpleMatrixUtil.h"
//#include "LinearSystem.h"
#include "Solver.h"
#include "EigenUtil.h"

//__align__(16)
//struct CorrPair  {
//  float3 src = make_float3(0);
//  float3 targ = make_float3(0);
//  float3 targNormal = make_float3(0);
//  float distance = 0; //between two correspondences
//  int dummy = -2; //padding
//};

//using thrust::device_vector;
//using thrust::device_ptr;

class CameraTracking  {

private:
  int width, height;
  //LinearSystem linearSystem;
  Solver solver;
  int maxIters = 30;
  float4* d_correspondenceNormals;
  float4* d_correspondences;
  float* d_residuals;
  //thrust::device_vector<CorrPair> d_coordPair;
  //device_vector<float4> d_correspondences;
  //device_vector<float4> d_correspondenceNormals;
  //device_vector<float> d_residuals;
  Matrix4x4f deltaTransform;
  float globalCorrespondenceError = 0.0f;
  Matrix4x4f delinearizeTransformation(const Vector6f & sol);
  Eigen::Matrix4f rigidAlignment(const float4*, const float4*, const Eigen::Matrix4f&);

public:

  CameraTracking(int, int);
  ~CameraTracking();
  void Align(float4*, float4*, float4*, float4*, const uint16_t*, const uint16_t*);
  Matrix4x4f getTransform() { return deltaTransform; }
};

#endif
