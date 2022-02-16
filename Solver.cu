#include <cuda_runtime_api.h>
#include <cublas_v2.h>
//#include "Solver.h"
#include "cuda_helper/helper_math.h"
#include <thrust/fill.h>
#include <thrust/device_ptr.h>

#define numCols 640

const int pyr_size = 3;
std::array<dim3, pyr_size> solver_blocks  = {dim3(20, 15), dim3(20, 15), dim3(20, 15)}; //, dim3(16, 12)};
std::array<dim3, pyr_size> solver_threads = {dim3(32, 32), dim3(16, 16), dim3(8,   8)}; //, dim3(5, 5)}	;

//using FloatVec = thrust::device_vector<float>;
//using Float4Vec = thrust::device_vector<float4>;

//using CorrPairVec = thrust::device_vector<CorrPair>;

__device__ inline
float CalculateResidual(const float3& n, const float3& d, const float3& s)
{
  float3 p = (d - s);
  return (dot(p,n));
}

__device__ inline
void CalculateJacobians(float* JacMat, const float3& d, const float3& n, int index)
{
  float3 T = (cross(d, n));
  // Calculate Jacobian for this correspondence pair. Probably most important piece
  // of code in entire project
  JacMat[index*6]     = n.x;
  JacMat[index*6 + 1] = n.y;
  JacMat[index*6 + 2] = n.z;
  JacMat[index*6 + 3] = T.x;
  JacMat[index*6 + 4] = T.y;
  JacMat[index*6 + 5] = T.z;
  //JacMat.row(index) << n.x, n.y, n.z, T.x, T.y, T.z ;
}

__global__
void CalculateJacAndResKernel(const float4* d_src, const float4* d_dest, const float4* d_destNormals,float* d_JacMat,
															const int width, const int height)
{
	int xidx = blockDim.x*blockIdx.x + threadIdx.x;
	int yidx = blockDim.y*blockIdx.y + threadIdx.y;
	//find globalIdx row-major
	const int idx = (yidx*width) + xidx;
  float3 src = make_float3(d_src[idx]);
  float3 dest = make_float3(d_dest[idx]);
  float3 destNormal = make_float3(d_destNormals[idx]);
  CalculateJacobians(d_JacMat, dest, destNormal, idx);
  //residual[idx] = pair.distance;
}

//__device__ inline
//void CalculateJTJ

extern "C" void CalculateJacobiansAndResiduals(const float4* d_src, const float4* d_targ, const float4* d_targNormals,
    float* d_Jac, float* d_residuals, const int pyrLevel, const int width, const int height)
{

  //First calculate Jacobian and Residual matrices
  //float4* d_targ = thrust::raw_pointer_cast(&targ[0]);
  //float4* d_targNormals = thrust::raw_pointer_cast(&targNormals[0]);
  //float* d_jacobianMatrix = thrust::raw_pointer_cast(&Jac[0]);
  //float* d_resVector = thrust::raw_pointer_cast(&residual[0]);
  //float* d_jtj = thrust::raw_pointer_cast(&JTJ[0]);
  //float* d_jtr = thrust::raw_pointer_cast(&JTr[0]);
	int numCorrPairs = width*height; //number of correspondence pairs
  thrust::device_ptr<float> d_Jac_ptr = thrust::device_pointer_cast(d_Jac);
  thrust::fill(d_Jac_ptr, d_Jac_ptr+(numCorrPairs*6), 0);  //TODO - is this redundant?
  CalculateJacAndResKernel<<<solver_blocks[pyrLevel], solver_threads[pyrLevel]>>>(d_src, d_targ, d_targNormals, d_Jac,
			width, height);

  //Then calculate Matrix-vector JTr and Matrix-matrix JTJ products
}

