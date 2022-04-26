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

using float4_pyramid = std::array<float4*, 3>;
using uint16_pyramid = std::array<uint16_t*, 3>;

extern "C" float computeCorrespondences(const float4* d_input, const float4* d_target, const float4* d_targetNormals, float4* corres, float4* corresNormals, float* residual, float4x4 deltaTransform, int width, int height, int pyrLevel);

extern "C" float gaussianBlur(const uint16_t* d_inputDepth, uint16_t* d_outputDepth, const int width, const int height);
extern "C" bool  GenerateImagePyramids(const vector<device_ptr<uint16_t>>& d_PyramidDepths,
																			 const vector<device_ptr<float4>>& d_PyramidVerts,
																			 const vector<device_ptr<float4>>& d_PyramidNormals);

//extern "C" float computeCorrespondences_pyramid(const float4* d_input, const float4* d_target, const float4* d_targetNormals, float4* corres, float4* corresNormals, float* residual, const float4x4 deltaTransform, const int width, const int height);

extern "C" bool SetCameraIntrinsic(const float* intrinsic, const float* invIntrinsic);
extern "C" bool SetGaussianKernel(const float* kernel);
//Takes device pointers, calculates correct position and normals
//extern "C" void preProcess(float4 *positions, float4* normals, const uint16_t *depth);

//extern "C" void Allocate();
//extern "C" void Deallocate();

//-------------------------------------------------------------------------------
//----Two versions----for float4, uint16_t-----------
bool CameraTracking::AllocImagePyramid(float4* baseLayer, std::vector<device_ptr<float4>>& toFill_pyr)
{
	//[1] Set (already computed) input as 1st level of pyramid
	device_ptr<float4> imgLvl_0 = thrust::device_pointer_cast(baseLayer); //image level 0
	toFill_pyr.push_back(imgLvl_0);

	//[2] Alloc other levels of pyramid
	for(int pyrLevel=1; pyrLevel < pyramid_size; ++pyrLevel)
	{
		int width  = pyramid_resolution[pyrLevel][0];
		int height = pyramid_resolution[pyrLevel][1];
		//get width, height for this level
		size_t LEVEL_NUMPIXELS = width * height;

		float4* d_tmpPtr = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_tmpPtr, LEVEL_NUMPIXELS*sizeof(float4)));
		checkCudaErrors(cudaDeviceSynchronize());
		if(d_tmpPtr == nullptr) return false;

		device_ptr<float4> d_ptr = thrust::device_pointer_cast(d_tmpPtr);
		checkCudaErrors(cudaMemset(d_tmpPtr, 0, LEVEL_NUMPIXELS*sizeof(float4)));
		checkCudaErrors(cudaDeviceSynchronize());
		//thrust::fill(d_ptr, d_ptr + LEVEL_NUMPIXELS, 0.0f); //this doesn't work cus this isn't a .cu file
		toFill_pyr.push_back(d_ptr);
	}
	return true;
}

bool CameraTracking::AllocImagePyramid(uint16_t* baseLayer, std::vector<device_ptr<uint16_t>>& toFill_pyr)
{
	//[1] Set (already computed) input as 1st level of pyramid
	device_ptr<uint16_t> imgLvl_0 = thrust::device_pointer_cast(baseLayer); //image level 0
	toFill_pyr.push_back(imgLvl_0);

	//[2] Alloc other levels of pyramid
	for(int pyrLevel=1; pyrLevel < pyramid_size; ++pyrLevel)
	{
		int width  = pyramid_resolution[pyrLevel][0];
		int height = pyramid_resolution[pyrLevel][1];
		//get width, height for this level
		size_t LEVEL_NUMPIXELS = width * height;

		uint16_t* d_tmpPtr = nullptr;
    checkCudaErrors(cudaMalloc((void**)&d_tmpPtr, LEVEL_NUMPIXELS*sizeof(uint16_t)));
		checkCudaErrors(cudaDeviceSynchronize());

		if(d_tmpPtr == nullptr) return false;

		device_ptr<uint16_t> d_ptr = thrust::device_pointer_cast(d_tmpPtr);
		checkCudaErrors(cudaMemset(d_tmpPtr, 0, LEVEL_NUMPIXELS*sizeof(uint16_t)));
		checkCudaErrors(cudaDeviceSynchronize());
		//thrust::fill(d_ptr, d_ptr + LEVEL_NUMPIXELS, 0); //this doesn't work cus this isn't a cu file
		toFill_pyr.push_back(d_ptr);
	}
	return true;
}

//bool GaussianBlurPyramid()
//{
//	//Blur kernel runs top-down, i.e process higher resolutions first and go decreasing order.
//	//[1] First raw depth
//	for(int pyrLevel = 0; pyrLevel < pyramid_size-1; ++pyrLevel)
//	{
//		const int width 										= pyramid_resolution[pyrLevel+1][0] ;
//		const int height 										= pyramid_resolution[pyrLevel+1][1] ;
//		const uint16_t* d_referenceDepthMap = thrust::raw_pointer_cast(inputDepthsPyr[pyrLevel]);
//		uint16_t* d_toFillDepthMap 					= thrust::raw_pointer_cast(inputDepthsPyr[pyrLevel+1]);
//		//GaussianBlur(d_referenceDepthMap, d_toFillDepthMap,
//	}
//
//}

void CameraTracking::Align(float4*   d_inputVerts,   float4* d_inputNormals,
													 float4*   d_targetVerts,  float4* d_targetNormals,
													 uint16_t* d_inputDepths,  uint16_t* d_targetDepths)
{
	bool status = true;

	//Alloc resources
	if(!pyramid_alloced)
	{
		status &=AllocImagePyramid(d_inputVerts,    d_inputVerts_pyr);
		status &=AllocImagePyramid(d_inputNormals,  d_inputNormals_pyr);
		status &=AllocImagePyramid(d_inputDepths, 	 d_inputDepths_pyr);

		status &=AllocImagePyramid(d_targetVerts,   d_targetVerts_pyr);
		status &=AllocImagePyramid(d_targetNormals, d_targetNormals_pyr);
		status &=AllocImagePyramid(d_targetDepths,  d_targetDepths_pyr);

		if (!status)
		 std::runtime_error("Failed to generate Image pyramids! ");
		else 
			pyramid_alloced = true;
	}

	std::cout << " Generating Image Pyramid : Input frame";
	status &= GenerateImagePyramids(d_inputDepths_pyr,  d_inputVerts_pyr,  d_inputNormals_pyr);
	std::cout << " Generating Image Pyramid : Target frame";
	status &= GenerateImagePyramids(d_targetDepths_pyr, d_targetVerts_pyr, d_targetNormals_pyr);

	if (!status)
   std::runtime_error("Failed to generate Image pyramids! ");

	int width  = -1; //numCols;
  int height = -1; //numRows;

	//GaussianBlurPyramid();
  //for (int iter = 0; iter < maxIters; iter++) {
  for (int pyrLevel = pyramid_size-1; pyrLevel >= 0; --pyrLevel)
  {
		//todo cleanup
		width  = pyramid_resolution[pyrLevel][0];
		height = pyramid_resolution[pyrLevel][1];

		std::cout << "\n" << termcolor::on_green << "PyrLevel : "<< pyrLevel << ", Resolution : "<< width <<" x  " << height << termcolor::reset << "\n";
		std::cout << termcolor::underline << "                                               \n" << termcolor::reset;

    for (int iter = 0; iter < pyramid_iters[pyrLevel]; iter++)
    {
      //globalCorrespondenceError = 0.0f;
      std::cout << "\n" << termcolor::on_red << "Iteration : " << iter << termcolor::reset << "\n";
      std::cout << termcolor::underline << "                                               \n" << termcolor::reset;

      //CUDA files cannot include any Eigen headers, don't know why. So convert eigen matrix to __device__ compatible float4x4.
      cout << "deltaTransform = \n" << deltaTransform << "\n";
      float4x4 deltaT = float4x4(deltaTransform.data());
      deltaT.transpose();

      //Clear previous data
      //TODO All move all this to .cu file
      //thrust::fill(d_correspondences.begin(), d_correspondences.end(), make_float4(0));
      //thrust::fill(d_correspondenceNormals.begin(), d_correspondenceNormals.end(), make_float4(0));
      //thrust::fill(d_residuals.begin(), d_residuals.end(), (float)0.0f);


      //We now have all data we need. Find correspondence pairs.
			float4* d_inputVerts_lvl = thrust::raw_pointer_cast(d_inputVerts_pyr[pyrLevel]);
			float4* d_targetVerts_lvl = thrust::raw_pointer_cast(d_targetVerts_pyr[pyrLevel]);
			float4* d_inputNormals_lvl = thrust::raw_pointer_cast(d_inputNormals_pyr[pyrLevel]);
			float4* d_targetNormals_lvl = thrust::raw_pointer_cast(d_targetNormals_pyr[pyrLevel]);

      globalCorrespondenceError = computeCorrespondences(d_inputVerts_lvl, d_targetVerts_lvl,
																												 d_targetNormals_lvl,
																												 d_tmpCorrespondences, 
																												 d_tmpCorrespondenceNormals,
																												 d_tmpResiduals, deltaT, width, 
																												 height, pyrLevel);
      std::cout << "\n" << termcolor::on_blue << termcolor::white << "globalCorrespondenceError = " << globalCorrespondenceError << termcolor::reset << "\n";

      //if (globalCorrespondenceError == 0.0f) {
      //  std::cout << "\n\n" << termcolor::bold << termcolor::grey << termcolor::on_white << "Correspondence error is zero. Stopping." << termcolor::reset << "\n\n";
      //  break;
      //}

      std::cout<<termcolor::green<<"Global correspondence error = "<<globalCorrespondenceError<<termcolor::reset<<" \n\n";
      //Matrix4x4f deltaT = Matrix4x4f(deltaTransform.data());

      solver.BuildLinearSystem(d_inputVerts_lvl, d_tmpCorrespondences, d_tmpCorrespondenceNormals, d_tmpResiduals, pyrLevel);
      Matrix4x4f intermediateT = solver.getTransform();
      //Matrix4x4f intermediateT = rigidAlignment(d_input, d_inputNormals, deltaTransform);
      deltaTransform = intermediateT;//intermediateT*deltaTransform;
      //float4x4 transposed = deltaTransform.transpose(); //TODO : remove later
    }
  }
}

/*
Extract a transform matrix from our solved vector
*/
Matrix4x4f CameraTracking::delinearizeTransformation(const Vector6f& sol)
{
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

/* Run a gaussian blur over input depth map */
void CameraTracking::GaussianBlur(const uint16_t* d_inputDepthMap, uint16_t* d_outputDepthMap, const int width, const int height)
{
	gaussianBlur(d_inputDepthMap, d_outputDepthMap, width, height);
}

bool CameraTracking::AllocTmpBuffers(float4* d_tmpCorrespondences, 
																		 float4* d_tmpCorrespondenceNormals ,
																		 float* d_tmpResiduals)
{
  bool status = false;
  return status;
}

CameraTracking::CameraTracking(int w, int h):
	g_width(w),
	g_height(h),
	pyramid_alloced(false),
	d_tmpResiduals(nullptr),
	d_tmpCorrespondences(nullptr),
	d_tmpCorrespondenceNormals(nullptr)
{
  const size_t F4_ARRAY_SIZE = w*h*sizeof(float4);
  const size_t ARRAY_SIZE    = w*h*sizeof(float);

  bool status = true;
	//alloc device buffers 
  checkCudaErrors(cudaMalloc((void**)&d_tmpCorrespondences, F4_ARRAY_SIZE));
	d_correspondences = thrust::device_pointer_cast(d_tmpCorrespondences);
	checkCudaErrors(cudaMemset(d_tmpCorrespondences, 0, F4_ARRAY_SIZE));
  if (!d_tmpCorrespondences)
    status = false;
	//thrust::fill(d_correspondences, d_correspondences + (w*h), 0);

  checkCudaErrors(cudaMalloc((void**)&d_tmpCorrespondenceNormals, F4_ARRAY_SIZE));
	d_correspondenceNormals = thrust::device_pointer_cast(d_tmpCorrespondenceNormals);
	checkCudaErrors(cudaMemset(d_tmpCorrespondenceNormals, 0, F4_ARRAY_SIZE));
	if (!d_tmpCorrespondenceNormals)
	  status = false;
	//thrust::fill(d_correspondenceNormals, d_correspondenceNormals + (w*h), 0);

  checkCudaErrors(cudaMalloc((void**)&d_tmpResiduals, ARRAY_SIZE));
	d_residuals = thrust::device_pointer_cast(d_tmpResiduals);
	checkCudaErrors(cudaMemset(d_tmpResiduals, 0, ARRAY_SIZE));
	if (!d_tmpResiduals)
	  status = false;

	if (!status)
   std::runtime_error("Allocation of temporary device vectors failed ");
	//thrust::fill(d_residuals, d_residuals + (w*h), 0);
	//init matrices
  float arr[16] = { 1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1 };
  deltaTransform = Matrix4x4f(arr);
  //const float intrinsics[] = {525.0, 0, 319.5, 0, 525.0, 239.5, 0, 0, 1}; //TODO: read from file
  Matrix3x3f K{intrinsics}; //Camera intrinsic matrix
  Matrix3x3f K_inv = K.inverse();
	//k_GaussianKernel = {1.0f, 0.0f, 0.0f,
	//											 0.0f, 1.0f, 0.0f,
	//											 0.0f, 0.0f, 1.0f};
	//k_GaussianKernel[] = {1.0/16.0f, 1.0/8.0f, 1.0/16.0f,
	//									  1.0/8.0f,  1.0/4.0f, 1.0/8.0f,
	//									  1.0/16.0f, 1.0/8.0f, 1.0/16.0f};
  std::cout<< termcolor::on_blue<< "Intrinsic camera matrix :" <<termcolor::reset<<"\n";
  std::cout<< termcolor::bold<< K << termcolor::reset<< "\n\n";

	Matrix3x3f gaussKernel{k_GaussianKernel};
  std::cout<< termcolor::on_blue<< "GaussianBlur kernel matrix :" <<termcolor::reset<<"\n";
  std::cout<< termcolor::bold<< gaussKernel << termcolor::reset<< "\n\n";

  SetCameraIntrinsic(K.data(), K_inv.data());
	SetGaussianKernel(k_GaussianKernel);
  //float4x4 transposed = deltaTransform.transpose();
}

CameraTracking::~CameraTracking()
{
  checkCudaErrors(cudaFree(d_tmpCorrespondences));
  checkCudaErrors(cudaFree(d_tmpCorrespondenceNormals));
  checkCudaErrors(cudaFree(d_tmpResiduals));
}


