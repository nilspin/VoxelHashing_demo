#include <cuda_runtime_api.h>
#include "cuda_helper/helper_cuda.h"
#include "cuda_helper/helper_math.h"
//#include "EigenUtil.h"

#define numRows 480
#define numCols 640
#define MINF __int_as_float(0xff800000)
#define MAXF __int_as_float(0x7F7FFFFF)

const int SYSTEM_SIZE = 27;

__device__ inline
bool isValid(float4 v) {
	return v.w != MINF;
}

__device__ inline
float calculate_b(float3 n, float3 d, float3 s) {
	return dot(n, d) - dot(n, s);
}

__global__
void buildLinearSystem(const float4* input, const float4* target, const float4* target_normal, float* linearSystem) {

	const int SYSTEM_SIZE = 27;	//we need 26 floats to build Ax=b system
	const int WINDOW_SIZE = 8;	//we process 8 correspondences per thread
	__shared__ extern float linearSystem_shared[];

	const int idx = (blockIdx.x * blockDim.x) + (WINDOW_SIZE * threadIdx.x);
	float4 s, d, n;
	for (int t = 0; t < WINDOW_SIZE; ++t) {
		s = input[idx + t];
		d = target[idx + t];
		n = target_normal[idx + t];

		float A[6];
		if (isValid(n)) {
			float b = calculate_b(make_float3(n), make_float3(d), make_float3(s));
			float3 first3 = cross(make_float3(s), make_float3(n));
			A[0] = first3.x;
			A[1] = first3.y;
			A[2] = first3.z;
			A[3] = n.x;
			A[4] = n.y;
			A[5] = n.z;
			//We now have enough information to build Ax=b system. Let's calculate ATA and ATb
			//printf("tid %d -- A : %f %f %f %f %f %f, b : %f \n", idx, A[0], A[1], A[2], A[3], A[4], A[5], b);
			int k = 0;
			for (int i = 0; i < 6; ++i) {
				for (int j = i; j < 6; ++j) {
					linearSystem_shared[SYSTEM_SIZE*threadIdx.x + (k++)] += A[i] * A[j];	//For ATA
				}
				linearSystem_shared[SYSTEM_SIZE*threadIdx.x + 21 + i] += A[i] * b;	//For ATb
			}
		}
	}
	__syncthreads();
	//Filled the 128*SYSTEM_SIZE matrix. Now reduce:
	for (uint s = 1; s < blockDim.x; s *= 2) {
		if (threadIdx.x % (2 * s) == 0) {
			for (int k = 0; k < SYSTEM_SIZE; ++k) {
				linearSystem_shared[SYSTEM_SIZE*threadIdx.x + k] += linearSystem_shared[SYSTEM_SIZE*(threadIdx.x + s) + k];
			}
		}
		__syncthreads();
	}

	//Whole system should've been reduced to 1st array by now.	

	if (threadIdx.x == 0) {
		//write block output to global memory
		for (int i = 0; i < SYSTEM_SIZE; ++i) {
			linearSystem[blockIdx.x*SYSTEM_SIZE + i] = linearSystem_shared[i];
		}
	}
}

extern "C" void buildLinearSystemOnDevice(const float4* d_input, const float4* d_target, const float4* d_targetNormals,
	float* d_generatedMatrixSystem, float* h_generatedMatrixSystem)
{
	dim3 blocks = dim3(300, 1, 1);
	dim3 threads = dim3(128);
	buildLinearSystem <<<blocks, threads, 128*SYSTEM_SIZE*sizeof(float) >>>(d_input, d_target, d_targetNormals, d_generatedMatrixSystem);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(h_generatedMatrixSystem, d_generatedMatrixSystem, blocks.x * SYSTEM_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

}