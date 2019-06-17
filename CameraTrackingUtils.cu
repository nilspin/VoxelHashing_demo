#ifndef CAMTRACKING_UTIL
#define CAMTRACKING_UTIL

#if defined(_WIN32)
  #include <Windows.h>
#endif

#include <iostream>
#include <cstdio>
#include <cuda_runtime_api.h>
#include "cuda_helper/helper_cuda.h"
#include "cuda_helper/helper_math.h"
//#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include "common.h"

//This is a simple vector library. Use this with CUDA instead of GLM.
#include "cuda_helper/cuda_SimpleMatrixUtil.h"

//#define MINF __int_as_float(0xff800000)
//#define MAXF __int_as_float(0x7F7FFFFF)
#define fx 525
#define fy 525
#define cx 319.5
#define cy 239.5
//Kinect v2 specific camera params
#define numCols 640
#define numRows 480

//const float distThres = 5.0f;
//const float normalThres = -1.0f;
//const float idealError = 0.0f;
//Since numCols = 640 and numRows = 480, we set blockDim according to 32x32 tile
dim3 blocks = dim3(20, 15, 1);
dim3 threads = dim3(32, 32, 1);

//using thrust::device_vector;
//using thrust::device_ptr;

__device__ __constant__ float3x3 K;  //Camera intrinsic matrix
__device__ __constant__ float3x3 K_inv;
__device__ float globalError;

//__device__ inline
//bool isValid(float4 v) {
//	return v.w != MINF;
//}

__global__
void calculateVertexPositions(float4* d_vertexPositions, const uint16_t* d_depthBuffer) {
	int xidx = blockDim.x*blockIdx.x + threadIdx.x;
	int yidx = blockDim.y*blockIdx.y + threadIdx.y;

	if (xidx >= numCols || yidx >= numRows) {
		return;
	}

	//find globalIdx row-major
	const int idx = (yidx*numCols) + xidx;

	const float w = 1.0f; //flag to tell whether this is valid vertex or not
	uint16_t d = d_depthBuffer[idx];
	float depth = d / 5000.0f; //5000 units = 1meter. We're now dealing in meters.
	//if (depth == 0) {
	//	w = 0.0f;
	//}

  float3 imageCoord = make_float3(xidx, yidx, 1.0);
  float3 point = K_inv*imageCoord*depth;
  //float4 vertex = make_float4(point.x, -point.y, -point.z, w);
  float4 vertex = make_float4(point.x, point.y, point.z, w);
  d_vertexPositions[idx] = vertex;
}

__global__
void calculateNormals(const float4* d_positions, float4* d_normals)
{
	int xidx = blockDim.x*blockIdx.x + threadIdx.x;
	int yidx = blockDim.y*blockIdx.y + threadIdx.y;

	if (xidx >= numCols || yidx >= numRows) {
		return;
	}

	//find globalIdx row-major
	const int idx = (yidx*numCols) + xidx;

	//d_normals[idx] = make_float4(MINF, MINF, MINF, MINF);
	d_normals[idx] = make_float4(0, 0, 0, 0);

	if (xidx > 0 && xidx < numCols - 1 && yidx > 0 && yidx < numRows - 1) {
		const float4 CC = d_positions[(yidx + 0)*numCols + (xidx + 0)];
		const float4 PC = d_positions[(yidx + 1)*numCols + (xidx + 0)];
		const float4 CP = d_positions[(yidx + 0)*numCols + (xidx + 1)];
		const float4 MC = d_positions[(yidx - 1)*numCols + (xidx + 0)];
		const float4 CM = d_positions[(yidx + 0)*numCols + (xidx - 1)];

		if (CC.x != 0 && PC.x != 0 && CP.x != 0 && MC.x != 0 && CM.x != 0)
		{
			const float3 n = cross(make_float3(PC) - make_float3(MC), make_float3(CP) - make_float3(CM));
			const float  l = length(n);

			if (l > 0.0f)
			{
				//float4 v = make_float4(n/-l, 1.0f);
				float4 vert = make_float4(n/l, 0.0);
				d_normals[idx] = vert;
				//printf("Normal for thread %d : %f %f %f", yidx*numRows+xidx, vert.x, vert.y, vert.z);
			}
		}
	}
}

extern "C" void preProcess(float4 *positions, float4* normals, const uint16_t *depth) {
	calculateVertexPositions <<<blocks, threads >>>(positions, depth);
	calculateNormals <<<blocks, threads >>>(positions, normals);
	checkCudaErrors(cudaDeviceSynchronize());

}

__device__
static inline int2 cam2screenPos(float3 p) {
  float3 sp = K*p;
  //return make_int2(sp.x + 0.5, sp.y + 0.5);
	//float x = ((p.x * fx) / p.z) + cx;
	//float y = ((p.y * fy) / p.z) + cy;
	return make_int2(sp.x/sp.z + 0.5, sp.y/sp.z + 0.5);
}

__global__
void FindCorrespondences(const float4* input,	const float4* target,
    const float4* targetNormals, float4* correspondences, float4* correspondenceNormals,
    float* residuals,	const float4x4 deltaT,
    float distThres, float normalThres, int width, int height)
{

  const int offset = 1;
	int xidx = blockDim.x*blockIdx.x + threadIdx.x;
	int yidx = blockDim.y*blockIdx.y + threadIdx.y;

  //if (threadIdx.x==0 && threadIdx.y ==0)  {
  //  printf("Block is (%i, %i)\n",blockIdx.x, blockIdx.y);
  //}
	if (xidx >= numCols || yidx >= numRows) {
		return;
	}

	const int idx = (yidx*width) + xidx;

	float4 pSrc = input[idx];

	if (pSrc.z != 0) {	//if both pos and normal are valid points
    pSrc.w = 1.0f;
    float4 transPSrc = deltaT * pSrc;

		int2 projected = cam2screenPos(make_float3(transPSrc));
    int2 &sp = projected;
    //sp.x = sp.x/offset;
    //sp.y = sp.y/offset;

    if(sp.x > 0 && sp.y > 0 && sp.x < width && sp.y < height)
    {
      //printf("%i) sp.x = %i
      int targetIndex = (sp.y * width) + sp.x;
      float4 pTar = target[targetIndex];
      float4 nTar = targetNormals[targetIndex];
      float3 diff = make_float3(transPSrc - pTar);
      float d = dot(diff, make_float3(nTar));
      if (d < distThres)  {
        //if (threadIdx.x ==0 && threadIdx.y ==0)
        {
          //printf("%i) src- (%f, %f, %f), target- (%f, %f, %f), d= %f\n",idx, pSrc.x, pSrc.y, pSrc.z, pTar.x, pTar.y, pTar.z, d);
        }
        atomicAdd(&globalError, d);
        correspondences[idx] = pTar;
        correspondenceNormals[idx] = nTar;
        residuals[idx] = d;
        //coordpairs[idx].srcindex = idx;
        //coordpairs[idx].targIndex = targetIndex;
        //coordpairs[idx].srcindex = d;
      }
    }
	}
}

extern "C" float computeCorrespondences(const float4* d_input, const float4* d_target,
    const float4* d_targetNormals, float4* corres,
    float4* corresNormals, float* residuals,
    const float4x4 deltaTransform, const int width, const int height)
{
	//First clear the previous correspondence calculation
  checkCudaErrors(cudaMemcpyToSymbol(globalError, &idealError, sizeof(float)));

  thrust::device_ptr<float4> corres_ptr = thrust::device_pointer_cast(corres);
  thrust::device_ptr<float4> corresNormals_ptr = thrust::device_pointer_cast(corresNormals);
  thrust::device_ptr<float> residuals_ptr = thrust::device_pointer_cast(residuals);

  //std::cerr<<"Before clearing prev correspondences\n";

  thrust::fill(corres_ptr, corres_ptr + (width*height), float4{0,0,0,0});
  thrust::fill(corresNormals_ptr, corresNormals_ptr+ (width*height), float4{0,0,0,0});
  thrust::fill(residuals_ptr, residuals_ptr+ (width*height), (float)0.0f);

  checkCudaErrors(cudaDeviceSynchronize());
  //std::cerr<<"After clearing prev correspondences\n";
	FindCorrespondences <<<blocks, threads>>>(d_input, d_target, d_targetNormals,
      corres, corresNormals, residuals,	deltaTransform, distThres, normalThres, width, height);
  checkCudaErrors(cudaDeviceSynchronize());

  float globalErrorReadback = 0.0;
  checkCudaErrors(cudaMemcpyFromSymbol(&globalErrorReadback, globalError, sizeof(float)));
  //std::cerr<<"Global correspondence error = "<<globalErrorReadback<<" \n\n";
	checkCudaErrors(cudaDeviceSynchronize());
  return globalErrorReadback;
}

extern "C" bool SetCameraIntrinsic(const float* intrinsic, const float* invIntrinsic) {
  checkCudaErrors(cudaMemcpyToSymbol(K, intrinsic, 9*sizeof(float)));
  checkCudaErrors(cudaMemcpyToSymbol(K_inv, invIntrinsic, 9*sizeof(float)));
  return true;
}
#endif // CAMTRACKING_UTIL
