#ifndef CAMERA_TRACKING_H
#define CAMERA_TRACKING_H

#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_CUDA
#define GLM_SWIZZLE

#if defined(_WIN32)
  #include <Windows.h>
#endif

#include <cuda_runtime_api.h>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

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

using thrust::device_vector;
using thrust::device_ptr;

class CameraTracking  {

private:
  int g_width  = 0;
	int g_height = 0;
	bool m_pyramid_alloced = false;

  Solver solver;

	//----raw handles to device_ptrs-----
	float4* d_tmpCorrespondences = nullptr;
	float4* d_tmpCorrespondenceNormals = nullptr;
	float* d_tmpResiduals = nullptr;
	//-----------------------------------

  device_ptr<float4> d_correspondenceNormals;
  device_ptr<float4> d_correspondences;
  device_ptr<float>  d_residuals;

  //----Image pyramid------------------
  vector<device_ptr<float4>>   d_inputVerts_pyr;
  vector<device_ptr<float4>>   d_inputNormals_pyr;
  vector<device_ptr<uint16_t>> d_inputDepths_pyr;

  vector<device_ptr<float4>>   d_targetVerts_pyr;
  vector<device_ptr<float4>>   d_targetNormals_pyr;
  vector<device_ptr<uint16_t>> d_targetDepths_pyr;

  vector<device_ptr<float4>>   d_tempVerts_pyr; //these are used for swapping
  vector<device_ptr<float4>>   d_tempNormals_pyr;
  vector<device_ptr<uint16_t>> d_tempDepths_pyr;

  //----Debug-Analogs of dev buffers above on host--
  vector<vector<float4>>               h_inputVerts_pyr;
  vector<vector<float4>>               h_inputNormals_pyr;
  vector<vector<uint16_t>>             h_inputDepths_pyr;

  vector<vector<float4>>               h_targetVerts_pyr;
  vector<vector<float4>>               h_targetNormals_pyr;
  vector<vector<uint16_t>>             h_targetDepths_pyr;

  //----Matrices-----------------------
  Matrix4x4f deltaTransform;
  Matrix4x4f globalTransform;
	float k_GaussianKernel[9] = {1.0f, 0.0f, 0.0f,
	          									 0.0f, 1.0f, 0.0f,
	          									 0.0f, 0.0f, 1.0f};
  float globalCorrespondenceError = 0.0f;

	//----Methods------------------------
  bool            AllocTmpBuffers(float4*, float4*, float*);
  Matrix4x4f      delinearizeTransformation(const Vector6f& solution);
  Eigen::Matrix4f rigidAlignment(const float4*, const float4*, const Eigen::Matrix4f&);

	void GaussianBlur(const uint16_t* d_inputDepthMap, uint16_t* d_outDepthMap, int width, int height);
	//void GaussianBlurPyramid();
	template <typename T>
	bool AllocImagePyramid(T*, vector<device_ptr<T>>&);

  //----Debug methods for copying back to host and manually inspecting shit----------
  template <typename T>
  bool CopyDeviceBufToHost(const vector<device_ptr<T>>& , vector<vector<T>>& );

  template<typename T>
  bool WriteVec2File(const vector<vector<T>>&, const std::string);

public:

  CameraTracking(int, int);
  ~CameraTracking();
  void Align(float4*,  float4*,  float4*,  float4*,  uint16_t*,  uint16_t*);
	void swapBuffers();
	bool AllocImagePyramids(float4*, float4*, uint16_t*, float4*, float4*, uint16_t*);
  Matrix4x4f getDeltaTransform() { return deltaTransform; }
  Matrix4x4f getGlobalTransform() { return globalTransform; }
};

#endif
