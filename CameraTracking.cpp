#if defined(_WIN32)
  #include <Windows.h>
#endif

#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
//#include <thrust/fill.h>
#include "CameraTracking.h"
#include "termcolor.hpp"
#include "DebugHelper.hpp"

extern "C" float computeCorrespondences(const float4* d_input, const float4* d_target, const float4* d_targetNormals, float4* corres, float4* corresNormals, float* residual, const float4x4 deltaTransform, const int width, const int height);

extern "C" bool SetCameraIntrinsic(const float* intrinsic, const float* invIntrinsic);
//Takes device pointers, calculates correct position and normals
extern "C" void preProcess(float4 *positions, float4* normals, const uint16_t *depth);

//extern "C" void Allocate();
//extern "C" void Deallocate();

//-------------------------------------------------------------------------------

void CameraTracking::Align(float4* d_input, float4* d_inputNormals, float4* d_target,
  float4* d_targetNormals, const uint16_t* d_depthInput, const uint16_t* d_depthTarget) {

  preProcess(d_input, d_inputNormals, d_depthInput);
  preProcess(d_target, d_targetNormals, d_depthTarget);
  //WriteDeviceArrayToFile(d_inputNormals, "sourceNormalsDevice", width*height);
  int width = numCols;
  int height = numRows;

  for (int iter = 0; iter < maxIters; iter++) {
    //globalCorrespondenceError = 0.0f;
    //std::cout<< "\n"<<termcolor::on_red<< "Iteration : "<<iter << termcolor::reset << "\n";
    //std::cout << termcolor::underline <<"                                               \n"<< termcolor::reset;

    //CUDA files cannot include any Eigen headers, don't know why. So convert eigen matrix to __device__ compatible float4x4.
    //cout<<"deltaTransform = \n"<<deltaTransform<<"\n";
	  float4x4 deltaT = float4x4(deltaTransform.data());
    deltaT.transpose();

    //Clear previous data
    //TODO All move all this to .cu file
    //thrust::fill(d_correspondences.begin(), d_correspondences.end(), make_float4(0));
    //thrust::fill(d_correspondenceNormals.begin(), d_correspondenceNormals.end(), make_float4(0));
    //thrust::fill(d_residuals.begin(), d_residuals.end(), (float)0.0f);


	  //We now have all data we need. find correspondence.
	  globalCorrespondenceError = computeCorrespondences(d_input, d_target, d_targetNormals, d_correspondences, d_correspondenceNormals, d_residuals, deltaT, width, height);

    if(globalCorrespondenceError == 0.0f) {
      std::cout<<"\n\n"<<termcolor::bold<<termcolor::grey<<termcolor::on_white<<"Correspondence error is zero. Stopping."<<termcolor::reset<<"\n\n";
      break;
    }

    //std::cout<<termcolor::green<<"Global correspondence error = "<<globalCorrespondenceError<<termcolor::reset<<" \n\n";
	  //Matrix4x4f deltaT = Matrix4x4f(deltaTransform.data());

    solver.BuildLinearSystem(d_input, d_correspondences, d_correspondenceNormals, d_residuals, width, height);
	  Matrix4x4f intermediateT = solver.getTransform();
	  //Matrix4x4f intermediateT = rigidAlignment(d_input, d_inputNormals, deltaTransform);
    deltaTransform = intermediateT;//intermediateT*deltaTransform;
    //float4x4 transposed = deltaTransform.transpose(); //TODO : remove later
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
	//res.block(0, 3, 3, 1) = t;
  res.block(0, 3, 3, 1) = t;

	return res;
}

Eigen::Matrix4f CameraTracking::rigidAlignment(const float4* d_input, const float4* d_inputNormals, const Eigen::Matrix4f& deltaT) {
	Matrix4x4f computedTransform = deltaT;
	Matrix6x6f ATA; Vector6f ATb;
	//linearSystem.build(d_input, d_correspondence, d_correspondenceNormals, 0.0f, 0.0f, width, height, ATA, ATb);

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
	//std::cout << termcolor::green<< "Calculated transform : \n"<<termcolor::reset;
  //std::cout << newTransform << "\n";

	return newTransform;
}

CameraTracking::CameraTracking(int w, int h):width(w),height(h)
{
  const int F4_ARRAY_SIZE = width*height*sizeof(float4);
  const int ARRAY_SIZE = width*height*sizeof(float);
  checkCudaErrors(cudaMalloc((void**)&d_correspondences, F4_ARRAY_SIZE));
  checkCudaErrors(cudaMemset(d_correspondences, 0, F4_ARRAY_SIZE));
  checkCudaErrors(cudaMalloc((void**)&d_correspondenceNormals, F4_ARRAY_SIZE));
  checkCudaErrors(cudaMemset(d_correspondenceNormals, 0, F4_ARRAY_SIZE));
  checkCudaErrors(cudaMalloc((void**)&d_residuals, ARRAY_SIZE));
  checkCudaErrors(cudaMemset(d_residuals, 0, ARRAY_SIZE));
  float arr[16] = { 1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1 };
  deltaTransform = Matrix4x4f(arr);
  //const float intrinsics[] = {525.0, 0, 319.5, 0, 525.0, 239.5, 0, 0, 1}; //TODO: read from file
  Matrix3x3f K{intrinsics}; //Camera intrinsic matrix
  Matrix3x3f K_inv = K.inverse();
  std::cout<< termcolor::on_blue<< "Intrinsic camera matrix :" <<termcolor::reset<<"\n";
  std::cout<< termcolor::bold<< K << termcolor::reset<< "\n\n";
  SetCameraIntrinsic(K.data(), K_inv.data());
  //float4x4 transposed = deltaTransform.transpose();
}

CameraTracking::~CameraTracking()
{
  checkCudaErrors(cudaFree(d_correspondences));
  checkCudaErrors(cudaFree(d_correspondenceNormals));
  checkCudaErrors(cudaFree(d_residuals));
}


