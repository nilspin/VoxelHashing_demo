#ifndef SOLVER_H
#define SOLVER_H

#include "common.h"
#include "EigenUtil.h"
#include "DebugHelper.hpp"
#include "cuda_helper/helper_cuda.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <array>
#include "SE3.h"

//#include <thrust/device_vector.h>
//#include <thrust/device_ptr.h>

using namespace Eigen;
//using thrust::device_vector;

class Solver {
  public:
    unsigned int numIters = 10;

    void BuildLinearSystem(const float4* , const float4* , const float4* , const float* , int);

    void PrintSystem();

    void SolveJacobianSystem(const Matrix6x6f& JTJ, const Vector6f& JTr);

    Solver();
    ~Solver();

    Matrix4x4f getTransform() {return SE3Exp(estimate);};
    double getError() {return TotalError;};
  private:
    int JAC_MAX_SIZE;
    int RES_MAX_SIZE;
    int JTJ_MAX_SIZE;
    int JTr_MAX_SIZE;
    const int num_vars_in_jac = 6;
    std::array<float,36> raw_JTJ_matrix;
    Vector6f update, estimate; //Will hold solution
    bool solution_exists = false;
    //Matrix4x4f deltaT;  //intermediate estimated transform
    Vector6f JTr;  //for cost func [(T*src - dest)^2]
    //MatrixXf Jac;
    //VectorXf residual;
    double TotalError;
    int numCorrPairs = 0;
    Matrix6x6f JTJ, JTJinv;
    void CalculateJacobians(MatrixXf& J, const vec3& destVert,
        const vec3& destNormal, int index);
    void ComputeJTJandJTr(const Vector6f& J, Vector6f& JTJ, Vector6f& JTr);
    Matrix4x4f DelinearizeTransform(const Vector6f& x);

    //CUDA stuff

		const int pyr_size                 = 3;
		std::array<dim3, 3> numBlocks  = {dim3(20, 15), dim3(20, 15), dim3(20, 15)}; //, dim3(16, 12)};
		std::array<dim3, 3> numThreads = {dim3(32, 32), dim3(16, 16), dim3(8,   8)}; //, dim3(5, 5)}	;
    float* 										 d_Jac 	 		= nullptr;
    float* 										 d_JTr 	 		= nullptr;
    float* 										 d_JTJ 	 		= nullptr;
    //thrust::device_vector<float> d_Jac;  //Computed on device
    //thrust::device_vector<float> d_residual; //Computed on device
    //thrust::device_vector<float> d_JTr;  //then multiplied on device
    //thrust::device_vector<float> d_JTJ;  //finally this is computed

    cudaError_t 							 cu_res;
    cublasStatus_t  					 cuB_res;
    cublasHandle_t  					 cuB_handle;

};


#endif
