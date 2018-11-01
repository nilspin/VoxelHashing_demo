#if defined(_WIN32)
  #include <Windows.h>
#endif

#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include "CameraTracking.h"
#include "termcolor.hpp"


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

  for (int iter = 0; iter < maxIters; iter++) {
    std::cout<< termcolor::on_red<< "Iteration : "<<iter<< termcolor::reset<< "\n";
	  //We now have all data we need. find correspondence.
	  float4x4 deltaT = float4x4(deltaTransform.data());

	  computeCorrespondences(d_input, d_inputNormals, d_target, d_targetNormals, d_correspondence, d_correspondenceNormals,
		  deltaT, width, height);

	  //Matrix4x4f deltaT = Matrix4x4f(deltaTransform.data());

	  Matrix4x4f partialTransform = rigidAlignment(d_input, d_inputNormals, deltaTransform);
	  deltaTransform = partialTransform;
  }
}

/*
Extract a transform matrix from our solved vector
*/
Matrix4x4f CameraTracking::delinearizeTransformation(const Vector6f& sol) {
	Matrix4x4f res; res.setIdentity();

	//Rotation
	Matrix3x3f R = Eigen::AngleAxisf(sol[0], Eigen::Vector3f::UnitZ()).toRotationMatrix()*
		Eigen::AngleAxisf(sol[1], Eigen::Vector3f::UnitY()).toRotationMatrix()*
		Eigen::AngleAxisf(sol[2], Eigen::Vector3f::UnitX()).toRotationMatrix();
	//Translation
	Eigen::Vector3f t = sol.segment(3, 3);

	res.block(0, 0, 3, 3) = R;
	res.block(0, 3, 3, 1) = t;

	return res;
}

Eigen::Matrix4f CameraTracking::rigidAlignment(const float4* d_input, const float4* d_inputNormals, const Eigen::Matrix4f& deltaT) {
	Matrix4x4f computedTransform = deltaT;
	Matrix6x6f ATA; Vector6f ATb;
	linearSystem.build(d_input, d_correspondence, d_correspondenceNormals, 0.0f, 0.0f, width, height, ATA, ATb);

	//solve 6x7 matrix of linear equations
	std::cout << termcolor::green <<"Filled matrix system ATA | ATb : \n"<< termcolor::reset;
  for(int i=0;i<6;++i)  {
    for(int j=0;j<6;++j)  {
      std::cout<<ATA(i,j) <<" ";
    }
    std::cout<< "| "<<ATb(i)<<"\n";
  }

	Eigen::JacobiSVD<Matrix6x6f> SVD(ATA, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Vector6f x = SVD.solve(ATb);

	//then delinearize the computed matrix to extract transform matrix
	Matrix4x4f newTransform = delinearizeTransformation(x);
	std::cout << termcolor::green<< "Calculated transform : \n"<<termcolor::reset;
  std::cout << newTransform << "\n";

	return newTransform;
}

CameraTracking::CameraTracking(int w, int h):width(w),height(h)
{
  const int ARRAY_SIZE = width*height*sizeof(float4);
  checkCudaErrors(cudaMalloc((void**)&d_correspondence, ARRAY_SIZE));
  checkCudaErrors(cudaMemset(d_correspondence, 0, ARRAY_SIZE));
  checkCudaErrors(cudaMalloc((void**)&d_correspondenceNormals, ARRAY_SIZE));
  checkCudaErrors(cudaMemset(d_correspondenceNormals, 0, ARRAY_SIZE));
  float arr[16] = { 1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1 };
  deltaTransform = Matrix4x4f(arr);
  //float4x4 transposed = deltaTransform.transpose();
}

CameraTracking::~CameraTracking()
{
  checkCudaErrors(cudaFree(d_correspondence));
  checkCudaErrors(cudaFree(d_correspondenceNormals));
}


