#ifndef CAMTRACKING_UTIL
#define CAMTRACKING_UTIL

#if defined(_WIN32)
  #include <Windows.h>
#endif

#include <iostream>
#include <vector>
#include <cstdio>
#include <cuda_runtime_api.h>
#include "cuda_helper/helper_cuda.h"
#include "cuda_helper/helper_math.h"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/memory.h>
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

using std::vector;
//const float distThres = 5.0f;
//const float normalThres = -1.0f;
//const float idealError = 0.0f;
const int pyr_size = 3;
//Depth input res = 640x480. I set kernel launch params manually as per frames at each level in pyramid
std::array<dim3, pyr_size> blocks  = {dim3(20, 15), dim3(20, 15), dim3(20, 15)}; //, dim3(16, 12)};
std::array<dim3, pyr_size> threads = {dim3(32, 32), dim3(16, 16), dim3(8,   8)}; //, dim3(5, 5)}	;

using thrust::device_vector;
using thrust::device_ptr;

__device__ __constant__ float3x3 K;  //Camera intrinsic matrix
__device__ __constant__ float3x3 K_inv;
__device__ __constant__ float kGaussianKernel[9];
__device__ float globalError;

//__device__ inline
//bool isValid(float4 v)
//{
//	return v.w != MINF;
//}

//getting bilinearFilt code from original Voxelhashing repo
inline __device__
float bilinearInterpolate(int x, int y, const uint16_t* d_depthInput, uint32_t inWidth, uint32_t inHeight)
{
	const int2 p00 = make_int2(x, y);
	const int2 p01 = p00 + make_int2(0.0f, 1.0f);
	const int2 p10 = p00 + make_int2(1.0f, 0.0f);
	const int2 p11 = p00 + make_int2(1.0f, 1.0f);

	const float alpha = x - p00.x;
	const float beta  = y - p00.y;

	float s0 = 0.0f; float w0 = 0.0f;
	if(p00.x < inWidth && p00.y < inHeight) { float v00 = d_depthInput[p00.y*inWidth + p00.x]; if(v00 != MINF) { s0 += (1.0f-alpha)*v00; w0 += (1.0f-alpha); } }
	if(p10.x < inWidth && p10.y < inHeight) { float v10 = d_depthInput[p10.y*inWidth + p10.x]; if(v10 != MINF) { s0 +=		 alpha *v10; w0 +=		 alpha ; } }

	float s1 = 0.0f; float w1 = 0.0f;
	if(p01.x < inWidth && p01.y < inHeight) { float v01 = d_depthInput[p01.y*inWidth + p01.x]; if(v01 != MINF) { s1 += (1.0f-alpha)*v01; w1 += (1.0f-alpha);} }
	if(p11.x < inWidth && p11.y < inHeight) { float v11 = d_depthInput[p11.y*inWidth + p11.x]; if(v11 != MINF) { s1 +=		 alpha *v11; w1 +=		 alpha ;} }

	const float p0 = s0/w0;
	const float p1 = s1/w1;

	float ss = 0.0f; float ww = 0.0f;
	if(w0 > 0.0f) { ss += (1.0f-beta)*p0; ww += (1.0f-beta); }
	if(w1 > 0.0f) { ss +=		beta *p1; ww +=		  beta ; }

	if(ww > 0.0f) return ss/ww;
	else		  return MINF;
}

__global__
void downscaleKernel(const uint16_t* d_refDepthMap, uint16_t* d_toFillDepthMap, int inWidth, int inHeight, int outWidth, int outHeight)
{
	//extern __shared__ float scratchBuf[];

	//todo : load data into float3x3 kernel

	int xidx = blockDim.x*blockIdx.x + threadIdx.x;
	int yidx = blockDim.y*blockIdx.y + threadIdx.y;
	//trueDim is grid without halo cells
	int trueDimx = blockDim.x - (3 - 1);
	int trueDimy = blockDim.y - (3 - 1);

  //if (threadIdx.x==0 && threadIdx.y ==0)  {
  //  printf("Block is (%i, %i)\n",blockIdx.x, blockIdx.y);
  //}
	if (xidx >= outWidth || yidx >= outHeight)
	{
		return;
	}

	const int idx = (yidx*outWidth) + xidx;

	d_toFillDepthMap[idx] = bilinearInterpolate(xidx, yidx, d_refDepthMap, inWidth, inHeight);
}

__global__
void calculateVertexPositions(float4* d_vertexPositions, const uint16_t* d_depthBuffer)
{
	int xidx = blockDim.x*blockIdx.x + threadIdx.x;
	int yidx = blockDim.y*blockIdx.y + threadIdx.y;

	int lvl_width  = blockDim.x * gridDim.x;
	int lvl_height = blockDim.y * gridDim.y;
	if (xidx >= lvl_width|| yidx >= lvl_height)
	{
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
  float4 vertex = make_float4(point.x, point.y, point.z, w);
  d_vertexPositions[idx] = vertex;
}

__global__
void calculateNormals(const float4* d_positions, float4* d_normals)
{
	int xidx = blockDim.x*blockIdx.x + threadIdx.x;
	int yidx = blockDim.y*blockIdx.y + threadIdx.y;

	int lvl_width  = blockDim.x * gridDim.x;
	int lvl_height = blockDim.y * gridDim.y;
	if (xidx >= lvl_width|| yidx >= lvl_height)
	{
		return;
	}

	//find globalIdx row-major
	const int idx = (yidx*lvl_width) + xidx;

	//d_normals[idx] = make_float4(MINF, MINF, MINF, MINF);
	d_normals[idx] = make_float4(0, 0, 0, 0);

	if (xidx > 0 && xidx < lvl_width - 1 && yidx > 0 && yidx < lvl_height - 1)
	{
		const float4 CC = d_positions[(yidx + 0)*lvl_width + (xidx + 0)];
		const float4 PC = d_positions[(yidx + 1)*lvl_width + (xidx + 0)];
		const float4 CP = d_positions[(yidx + 0)*lvl_width + (xidx + 1)];
		const float4 MC = d_positions[(yidx - 1)*lvl_width + (xidx + 0)];
		const float4 CM = d_positions[(yidx + 0)*lvl_width + (xidx - 1)];

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

extern "C" void generatePositionAndNormals(float4 *positions, float4* normals, const uint16_t *depth)
{
	calculateVertexPositions <<<blocks[0], threads[0] >>>(positions, depth);
	calculateNormals <<<blocks[0], threads[0] >>>(positions, normals);
	checkCudaErrors(cudaDeviceSynchronize());
}

__device__
static inline int2 cam2screenPos(float3 p)
{
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
	if (xidx >= width || yidx >= height) //todo - should be based on mip-level width/height
	{
		return;
	}

	const int idx = (yidx*width) + xidx;

	float4 pSrc = input[idx];

	if (pSrc.z != 0)
	{
		//if both pos and normal are valid points
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
      if (d < distThres)
      {
        if (threadIdx.x ==0 && threadIdx.y ==0)
        {
          printf("%i) src- (%f, %f, %f), target- (%f, %f, %f), d= %f\n",idx, pSrc.x, pSrc.y, pSrc.z, pTar.x, pTar.y, pTar.z, d);
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
    float4x4 deltaTransform, int width, int height, int pyrLevel)
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
	FindCorrespondences <<<blocks[pyrLevel], threads[pyrLevel]>>>(d_input, d_target, d_targetNormals,
      corres, corresNormals, residuals,	deltaTransform, distThres, normalThres, width, height);
  checkCudaErrors(cudaDeviceSynchronize());

  float globalErrorReadback = 0.0;
  checkCudaErrors(cudaMemcpyFromSymbol(&globalErrorReadback, globalError, sizeof(float)));
  //std::cerr<<"Global correspondence error = "<<globalErrorReadback<<" \n\n";
	checkCudaErrors(cudaDeviceSynchronize());
  return globalErrorReadback;
}

extern "C" bool SetCameraIntrinsic(const float* intrinsic, const float* invIntrinsic)
{
  checkCudaErrors(cudaMemcpyToSymbol(K,     intrinsic,    9*sizeof(float)));
  checkCudaErrors(cudaMemcpyToSymbol(K_inv, invIntrinsic, 9*sizeof(float)));
  return true;
}

extern "C" bool SetGaussianKernel(const float* blurKernel)
{
  checkCudaErrors(cudaMemcpyToSymbol(kGaussianKernel, blurKernel, 9*sizeof(float)));
  return true;
}

extern "C" bool GenerateImagePyramids(const vector<device_ptr<uint16_t>>& d_PyramidDepths,
																			const vector<device_ptr<float4>>& d_PyramidVerts,
																			const vector<device_ptr<float4>>& d_PyramidNormals)
{
	//Blur kernel runs top-down, i.e process higher resolutions first and go decreasing order.
	//[1] First raw depth
  int inWidth  = -1, outWidth  = -1;
	int inHeight = -1, outHeight = -1;

	int inPyrLvl  = -1;
	int outPyrLvl = -1;
	//for(int pyrLevel = pyramid_size-1; pyrLevel > 0; --pyrLevel)
	for(int pyrLevel = 1; pyrLevel < pyramid_size-1; ++pyrLevel)
	{
    inPyrLvl                    = pyrLevel - 1;
    outPyrLvl                   = pyrLevel;

		inWidth 												= pyramid_resolution[inPyrLvl][0] ;
		inHeight 												= pyramid_resolution[inPyrLvl][1] ;
		uint16_t* d_referenceDepthMap   = thrust::raw_pointer_cast(d_PyramidDepths[inPyrLvl]);

		outWidth 												= pyramid_resolution[outPyrLvl][0] ;
		outHeight 											= pyramid_resolution[outPyrLvl][1] ;
		uint16_t* d_toFillDepthMap 		  = thrust::raw_pointer_cast(d_PyramidDepths[outPyrLvl]);

		downscaleKernel<<<blocks[pyrLevel], threads[pyrLevel]>>>(d_referenceDepthMap, d_toFillDepthMap, inWidth, inHeight, outWidth, outHeight);
		checkCudaErrors(cudaDeviceSynchronize());
	}

	//[2] and reconstruct camera-space vertices and normals
	//for(int pyrLevel = pyramid_size-1; pyrLevel > 0; --pyrLevel)
	for(int pyrLevel = 1; pyrLevel < pyramid_size; ++pyrLevel)
	{

		inWidth 											  = pyramid_resolution[pyrLevel][0] ;
		inHeight 											  = pyramid_resolution[pyrLevel][1] ;

		uint16_t* d_referenceDepthMap   = thrust::raw_pointer_cast(d_PyramidDepths[pyrLevel]);
		float4* d_toFillVertMap					= thrust::raw_pointer_cast(d_PyramidVerts[pyrLevel]);
		float4* d_toFillNormalMap				= thrust::raw_pointer_cast(d_PyramidNormals[pyrLevel]);

		calculateVertexPositions <<<blocks[pyrLevel], threads[pyrLevel] >>>(d_toFillVertMap, d_referenceDepthMap);
		checkCudaErrors(cudaDeviceSynchronize());

		calculateNormals <<<blocks[pyrLevel], threads[pyrLevel] >>>(d_toFillVertMap, d_toFillNormalMap);
		checkCudaErrors(cudaDeviceSynchronize());
	}

	return true;
}

__global__
void gaussianBlurKernel(const uint16_t* d_depthBufferIn, uint16_t* d_depthBufferOut, const int width, const int height, const dim3 tileSize)
{
	extern __shared__ float scratchBuf[];

	//todo : load data into float3x3 kernel

	int xidx = blockDim.x*blockIdx.x + threadIdx.x;
	int yidx = blockDim.y*blockIdx.y + threadIdx.y;
	//trueDim is grid without halo cells
	int trueDimx = blockDim.x - (3 - 1);
	int trueDimy = blockDim.y - (3 - 1);

  //if (threadIdx.x==0 && threadIdx.y ==0)  {
  //  printf("Block is (%i, %i)\n",blockIdx.x, blockIdx.y);
  //}
	if (xidx >= width || yidx >= height)
	{
		return;
	}

	const int idx = (yidx*width) + xidx;

	//now load a tile into scratchBuf
	scratchBuf[threadIdx.y*tileSize.x + threadIdx.x] = d_depthBufferIn[idx];

	__syncthreads();

	//..and perform blur
	float acc = 0.0f;
	if(threadIdx.y < trueDimy && threadIdx.x < trueDimx)
	{
		for(int i=0;i<3;++i)
		{
			for(int j=0;j<3;++j)
			{
				const int tileIdx = (blockDim.x * (threadIdx.y + i)) + (threadIdx.x + j);
				acc = scratchBuf[tileIdx];
				acc *= kGaussianKernel[(i*3) + j];
			}
		}
	}
	else
	{
		scratchBuf[threadIdx.y*tileSize.x + threadIdx.x] = 0.0f;
	}

	//write to output
	d_depthBufferOut[idx] = d_depthBufferIn[idx];
}

extern "C" void gaussianBlur(const uint16_t* d_depthBufferIn, uint16_t* d_depthBufferOut, const int width, const int height)
{
  checkCudaErrors(cudaDeviceSynchronize());
	for(int lvl=0;lvl<pyramid_size;++lvl)
	{
		const auto blockSize = blocks[lvl];
		const auto tileSize  = threads[lvl];
		gaussianBlurKernel<<<
			  blockSize, tileSize, (tileSize.x * tileSize.y *sizeof(float))
			>>>(d_depthBufferIn, d_depthBufferOut, width, height, tileSize); //use shared memory the size of a threadblock
	}
  checkCudaErrors(cudaDeviceSynchronize());
}

#endif // CAMTRACKING_UTIL
