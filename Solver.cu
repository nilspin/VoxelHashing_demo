#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include "Solver.h"

#define numCols 640
#define numRows 480
//Since numCols = 640 and numRows = 480, we set blockDim according to 32x32 tile
dim3 blocks = dim3(20, 15, 1);
dim3 threads = dim3(32, 32, 1);

using FloatVec = thrust::device_vector<float>;
using CorrPairVec = thrust::device_vector<CorrPair>;

__device__ inline
float CalculateResidual(const float3& n, const float3& d, const float3& s)  {
  float3 p = make_float3(d - s);
  return make_float(dot(p,n));
}

__global__
void CalculateJacAndResKernel(const CorrPair* correspondencePairs, float* JacMat, float* residual) {

	int xidx = blockDim.x*blockIdx.x + threadIdx.x;
	int yidx = blockDim.y*blockIdx.y + threadIdx.y;
	//find globalIdx row-major
	const int idx = (yidx*numCols) + xidx;
  CorrPair pair = correspondencePairs[idx];
  CalculateJacobians(JacMat, pair.targ, pair.targNormal, idx);
  residual[idx] = pair.distance;
}

__device__ inline
void CalculateJacobians(float* JacMat, const float3& d, const float3& n, int index)  {
  float3 T = make_float3(cross(d, n));
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

//__device__ inline
//void CalculateJTJ

extern "C" void CalculateJacobiansAndResidual(const CorrPairVec correspondencePairs, FloatVec Jac, FloatVec residual,
    FloatVec JTr, FloatVec JTJ) {

  //First calculate Jacobian and Residual matrices
  float* d_jacobianMatrix = thrust::raw_pointer_cast(&Jac[0]);
  float* d_resVector = thrust::raw_pointer_cast(&residual[0]);
  float* d_jtj = thrust::raw_pointer_cast(&JTJ[0]);
  float* d_jtr = thrust::raw_pointer_cast(&JTr[0]);
  CalculateJacAndResKernel<<<blocks, threads>>>(

  //Then calculate Matrix-vector JTr and Matrix-matrix JTJ products
}

