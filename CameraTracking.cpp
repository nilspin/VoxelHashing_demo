#include <Windows.h>

#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include "CameraTracking.h"


extern "C" void computeCorrespondences(const float4* d_input, const float4* d_inputNormals, const float4* d_target, const float4* d_targetNormals, float4* d_correspondence, float4* d_correspondenceNormals,
	const float4x4 deltaTransform, const int width, const int height);

//Takes device pointers, calculates correct position and normals
extern "C" void preProcess(float4 *positions, float4* normals, const uint16_t *depth);

//extern "C" void Allocate();
//extern "C" void Deallocate();

//-------------------------------------------------------------------------------

void CameraTracking::Align(float4* d_input, float4* d_inputNormals, float4* d_target, 
  float4* d_targetNormals, const uint16_t* d_depthInput, const uint16_t* d_depthTarget) {

  preProcess(d_input, d_inputNormals, d_depthInput);
  preProcess(d_target, d_targetNormals, d_depthTarget);

  //We now have all data we need. find correspondence.
  //glm::mat4 deltaT = mat4(1);
  
  computeCorrespondences(d_input, d_inputNormals, d_target, d_targetNormals, d_correspondence, d_correspondenceNormals, 
  					deltaTransform, width, height);
  
  Matrix4x4f deltaT = Matrix4x4f(deltaTransform.ptr());
  
  Matrix4x4f updatedDeltaT = rigidAlignment(d_input, d_inputNormals, deltaT);
   /*std::vector<float4> outputCUDA(640*480);
   std::cout<<"outputCuda.size()"<<outputCUDA.size()<<std::endl;
   checkCudaErrors(cudaMemcpy(outputCUDA.data(), d_correspondence, 640*480*sizeof(float4), cudaMemcpyDeviceToHost));
   std::ofstream fout("correspondenceData.txt");
   std::for_each(outputCUDA.begin(), outputCUDA.end(), [&fout](const float4 &n){fout<<n.x<<" "<<n.y<<" "<<n.z<<" "<<n.w<<"\n";});
   fout.close();
   */
}

Eigen::Matrix4f CameraTracking::rigidAlignment(const float4* d_input, const float4* d_inputNormals, const Eigen::Matrix4f& deltaT) {
	Matrix4x4f computedTransform = deltaT;
	Matrix6x7f system;
	linearSystem.build(d_input, d_correspondence, d_correspondenceNormals, 0.0f, 0.0f, deltaT, width, height, system);

	//solve 6x6 matrix of linear equations

	//then linearize the computed matrix to extract Rotation and Translation
	return computedTransform;
}

CameraTracking::CameraTracking(int w, int h):width(w),height(h)
{
  const int ARRAY_SIZE = width*height*sizeof(float4);
  checkCudaErrors(cudaMalloc((void**)&d_correspondence, ARRAY_SIZE));
  checkCudaErrors(cudaMemset(d_correspondence, 0, ARRAY_SIZE));
  checkCudaErrors(cudaMalloc((void**)&d_correspondenceNormals, ARRAY_SIZE));
  checkCudaErrors(cudaMemset(d_correspondenceNormals, 0, ARRAY_SIZE));
  float arr[16] = { 1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1 };
  deltaTransform = float4x4(arr);
  float4x4 transposed = deltaTransform.getTranspose();
}

CameraTracking::~CameraTracking()
{
  checkCudaErrors(cudaFree(d_correspondence));
  checkCudaErrors(cudaFree(d_correspondenceNormals));
}


