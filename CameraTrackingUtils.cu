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

const int pyr_size = 3;
//Depth input res = 640x480. I set kernel launch params manually as per frames at each level in pyramid
std::array<dim3, pyr_size> blocks  = {dim3(20, 15), dim3(20, 15), dim3(20, 15)}; //, dim3(16, 12)};
std::array<dim3, pyr_size> threads = {dim3(32, 32), dim3(16, 16), dim3(8,   8)}; //, dim3(5, 5)}	;

using thrust::device_vector;
using thrust::device_ptr;

__device__ __constant__ float3x3 K;  //Camera intrinsic matrix
__device__ __constant__ float3x3 K_inv;
__device__ __constant__ float kGaussianKernel[9];
__device__ float d_globalError;
__device__ unsigned int d_numCorresPairs;

//__device__ inline
//bool isValid(float4 v)
//{
//	return v.w != MINF;
//}

__device__
inline
uint16_t averagePixels(uint32_t x, uint32_t y, const uint16_t* d_depthInput, uint32_t inWidth, uint32_t inHeight)
{
  int2 curr				 = make_int2(x, y);
  int2 right			 = make_int2(curr.x + 1, curr.y + 0);
  int2 bottom			 = make_int2(curr.x + 0, curr.y + 1);
  int2 bottomright = make_int2(curr.x + 1, curr.y + 1);

	if (right.x >= inWidth || bottom.y >= inHeight)
	{
    return;
	}

  int idx1 = (curr.y * inWidth) + curr.x;
  int idx2 = (right.y * inWidth) + right.x;
  int idx3 = (bottom.y * inWidth) + bottom.x;
  int idx4 = (bottomright.y * inWidth) + bottomright.x;

	uint16_t d1 = d_depthInput[idx1];
	uint16_t d2 = d_depthInput[idx2];
	uint16_t d3 = d_depthInput[idx3];
	uint16_t d4 = d_depthInput[idx4];

	int sum = d1;
  sum += d2;
  sum += d3;
  sum += d4;

	int avg = sum / 4;

	uint16_t ret = avg;
  return ret;
}

/*
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
	if(p00.x < inWidth && p00.y < inHeight)
	{
		uint16_t v00 = d_depthInput[p00.y*inWidth + p00.x];
		if(v00 != MINF)
		{
			s0 += (1.0f-alpha)*v00;
			w0 += (1.0f-alpha);
		}
	}
	if(p10.x < inWidth && p10.y < inHeight)
	{
		uint16_t v10 = d_depthInput[p10.y*inWidth + p10.x];
		if(v10 != MINF)
		{
			s0 +=		 alpha *v10;
			w0 +=		 alpha ;
		}
	}

	float s1 = 0.0f; float w1 = 0.0f;
	if(p01.x < inWidth && p01.y < inHeight)
	{
		uint16_t v01 = d_depthInput[p01.y*inWidth + p01.x];
		if(v01 != MINF)
		{
			s1 += (1.0f-alpha)*v01;
		  w1 += (1.0f-alpha);
		}
	}

	if(p11.x < inWidth && p11.y < inHeight)
	{
		uint16_t v11 = d_depthInput[p11.y*inWidth + p11.x];
		if(v11 != MINF)
		{
			s1 +=		 alpha *v11;
			w1 +=		 alpha ;
		}
	}

	const float p0 = s0/w0;
	const float p1 = s1/w1;

	float ss = 0.0f; float ww = 0.0f;
	if(w0 > 0.0f)
	{
		ss += (1.0f-beta)*p0;
		ww += (1.0f-beta);
	}
	if(w1 > 0.0f)
	{
		ss +=		beta *p1;
		ww +=		  beta ;
	}

	if (ww > 0.0f)
  {
    float ret = ss / ww;
    return ret;
  }
  else
    return MINF;
}
*/

inline __device__
float bilinearInterpolate(int x, int y, const uint16_t* d_depthInput, uint32_t inWidth, uint32_t inHeight)
{
	const int2 p00 = make_int2(x, y);
	const int2 p01 = p00 + make_int2(0, 1);
	const int2 p10 = p00 + make_int2(1, 0);
	const int2 p11 = p00 + make_int2(1, 1);

	//const float alpha = x - p00.x; 0
	//const float beta  = y - p00.y; 0
  const float alpha = 0.0f;
  const float beta  = 0.0f;

	int s0 = 0; float w0 = 0.0f;
	if(p00.x < inWidth && p00.y < inHeight)
	{
		uint16_t v00 = d_depthInput[p00.y*inWidth + p00.x];
		{
			s0 += v00;
			w0 += 1.0f;
		}
	}
	if(p10.x < inWidth && p10.y < inHeight)
	{
		uint16_t v10 = d_depthInput[p10.y*inWidth + p10.x];
		{
			s0 +=	v10;
			w0 +=	1.0f ;
		}
	}

	int s1 = 0; float w1 = 0.0f;
	if(p01.x < inWidth && p01.y < inHeight)
	{
		uint16_t v01 = d_depthInput[p01.y*inWidth + p01.x];
		{
			s1 += v01;
		  w1 += 1.0f;
		}
	}

	if(p11.x < inWidth && p11.y < inHeight)
	{
		uint16_t v11 = d_depthInput[p11.y*inWidth + p11.x];
		{
			s1 +=	v11;
			w1 += 1.0f ;
		}
	}

	const float p0 = s0/w0;
	const float p1 = s1/w1;

	float ret = (p0 + p1) / 2.0f;
	return ret;

}

inline __device__
float gaussR(float sigma, float dist)
{
	return exp(-(dist*dist)/(2.0*sigma*sigma));
}

inline __device__
float  gaussD(float sigma, int x, int y)
{
	return exp(-((x*x+y*y)/(2.0f*sigma*sigma)));
}

inline __device__
float bilinearInterpolateDepth(const uint16_t* d_depthInput, uint16_t* d_depthOutput, uint32_t inWidth, uint32_t inHeight)
{
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
	int kernelRadius = (int)ceil(2.0*sigmaD);

	float sum = 0.0f;
	float sumWeight = 0.0f;

	float intCenter = input[dTid.xy];
	if(intCenter != MINF)
	{
		for(int m = xidx-kernelRadius; m <= xidx+kernelRadius; m++)
		{
			for(int n = yidx-kernelRadius; n <= yidx+kernelRadius; n++)
			{
				if(m >= 0 && n >= 0 && m < inWidth && n < inHeight)
				{
					int pos = (yidx*inWidth) + xidx;
					float intKerPos = d_depthInput[pos];

					if(intKerPos != MINF)
					{
						float weight = gaussD(sigmaD, m-dTid.x, n-dTid.y)*gaussR(sigmaR, intKerPos-intCenter);

						sumWeight += weight;
						sum += weight*intKerPos;
					}
				}
			}
		}

		if(sumWeight > 0.0f)
		{
			output[dTid.xy] = sum / sumWeight;
		}
	}
}

__global__
void downscaleKernel(const float4* d_refVertexMap, float4* d_toFillVertexMap, int inWidth, int inHeight, int outWidth, int outHeight, int offset)
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

	const int scatter_idx = (yidx*outWidth) + xidx;
  const int sample_idx  = ((yidx * numCols) + xidx) * offset;
  //const int sample_idx  = ((yidx * inWidth) + xidx) * offset;

	//uint16_t interpolated = bilinearInterpolate(xidx, yidx, d_refVertexMap, inWidth, inHeight);
	//uint16_t interpolated = averagePixels(xidx, yidx, d_refVertexMap, inWidth, inHeight);
	//d_toFillVertexMap[idx] = interpolated;
  float4 val                     = d_refVertexMap[sample_idx];
  d_toFillVertexMap[scatter_idx] = val;
}

__global__
void calculateVertexPositions(float4* d_vertexPositions, const uint16_t* d_depthBuffer)
{
	if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
	{
    printf("calculateVertPositions : Verts = %p \n depthBuff = %p\n", (void*)d_vertexPositions, (void*)d_depthBuffer);
	}
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
	if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
	{
    printf("calculateNormals : Verts = %p \n Normals = %p\n", (void*)d_positions, (void*)d_normals);
	}
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
static inline int2 cam2screenPos(float3 p, int offset)
{
  float3 sp = K*p;
  //return make_int2(sp.x + 0.5, sp.y + 0.5);
	//float x = ((p.x * fx) / p.z) + cx;
	//float y = ((p.y * fy) / p.z) + cy;
	//return make_int2(sp.x/sp.z + 0.5, sp.y/sp.z + 0.5);
  float2 tmp = make_float2(sp.x / (sp.z * offset) , sp.y / (sp.z * offset) );
	return make_int2(tmp.x, tmp.y);
}

__global__
void FindCorrespondences(const float4* input,	const float4* inputNormals,
		const float4* target, const float4* targetNormals, float4* correspondences,
		float4* correspondenceNormals, float* residuals,	const float4x4 deltaT,
    float corresThres, float normalThreshold, int width, int height, int pyrLevel)
{
	if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0)
	{
    printf("inputVerts = %p \n targetVerts = %p\n", (void*)input, (void*)target);
	}
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
  float4 nSrc = inputNormals[idx];

	if (pSrc.z != 0)
	{
		//if both pos and normal are valid points
    pSrc.w = 1.0f;
    float4 transPSrc = deltaT * pSrc;

    int offset = pow(2, pyrLevel); //scale projected screen-pos by pyramidLevel
		int2 projected = cam2screenPos(make_float3(transPSrc), offset);

    int2 &sp = projected;
		//sp.x /= divisor;
    //sp.y /= divisor;
    //sp.x = sp.x/offset;
    //sp.y = sp.y/offset;
		if ((sp.x > width) || (sp.y > height))
		{
      //printf("Projected pixel (%i, %i) out of bounds\n", sp.x, sp.y);
      return;
		}

    if(sp.x >= 0 && sp.y >= 0 && sp.x < width && sp.y < height)
    {
      //printf("%i) sp.x = %i
      int targetIndex = (sp.y * width) + sp.x;
      float4 pTar = target[targetIndex];
			//printf("%i | input idx : (%i, %i) val=(%f, %f, %f), target idx: (%i, %i) val=(%f, %f, %f)\n", idx, xidx, yidx, pSrc.x, pSrc.y, pSrc.z, projected.x, projected.y, pTar.x, pTar.y, pTar.z);
      float4 nTar = targetNormals[targetIndex];
      float3 diff = make_float3(transPSrc - pTar);
      float d = dot(diff, make_float3(nTar));

			//how 'similarly-pointing' are the normals for src and target?
      float  normalsDeviation = dot(nTar, nSrc);
      if ((d < corresThres) && ((normalsDeviation) > normalThreshold))
      {
        /*
        if (threadIdx.x ==0 && threadIdx.y ==0)
        {
          printf("%i) src- (%f, %f, %f), target- (%f, %f, %f), d= %f\n",idx, pSrc.x, pSrc.y, pSrc.z, pTar.x, pTar.y, pTar.z, d);
        }
				*/
        atomicAdd(&d_globalError, d);
        atomicAdd(&d_numCorresPairs, 1);

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

extern "C" float computeCorrespondences(const float4* d_input, const float4* d_inputNormals,
		const float4* d_target, const float4* d_targetNormals,
		float4* corres, float4* corresNormals, float* residuals,
    float4x4 deltaTransform, int width, int height, int pyrLevel,
	  float corresThres, float normalThreshold,
		unsigned int& h_numCorresPairs)
{
	//First clear the previous correspondence calculation
  checkCudaErrors(cudaMemcpyToSymbol(d_globalError, &ZERO_FLOAT, sizeof(float)));
  checkCudaErrors(cudaMemcpyToSymbol(d_numCorresPairs, &ZERO_UINT, sizeof(unsigned int)));

  thrust::device_ptr<float4> corres_ptr = thrust::device_pointer_cast(corres);
  thrust::device_ptr<float4> corresNormals_ptr = thrust::device_pointer_cast(corresNormals);
  thrust::device_ptr<float> residuals_ptr = thrust::device_pointer_cast(residuals);

  std::cerr<<"Before clearing prev correspondences\n";

  thrust::fill(corres_ptr, corres_ptr + (width*height), float4{0,0,0,0});
  thrust::fill(corresNormals_ptr, corresNormals_ptr+ (width*height), float4{0,0,0,0});
  thrust::fill(residuals_ptr, residuals_ptr+ (width*height), (float)0.0f);

  checkCudaErrors(cudaDeviceSynchronize());

  std::cerr<<"After clearing prev correspondences\n";

	FindCorrespondences <<<blocks[pyrLevel], threads[pyrLevel]>>>(d_input, d_inputNormals,
			d_target, d_targetNormals, corres, corresNormals, residuals,	deltaTransform, corresThres,
			normalThreshold, width, height, pyrLevel);
  checkCudaErrors(cudaDeviceSynchronize());

  float globalErrorReadback = 0.0;
	h_numCorresPairs = 0;
  checkCudaErrors(cudaMemcpyFromSymbol(&globalErrorReadback, d_globalError, sizeof(float)));
  checkCudaErrors(cudaMemcpyFromSymbol(&h_numCorresPairs, d_numCorresPairs, sizeof(unsigned int)));
	checkCudaErrors(cudaDeviceSynchronize());
  std::cerr<<"Total correspondence pairs = "<<h_numCorresPairs<<" \n\n";
  std::cerr<<"Correspondence pair % = "<<((float)h_numCorresPairs*100)/((float)width*height)<<"% \n\n";
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

extern "C" bool FillImagePyramids(vector<device_ptr<uint16_t>>& d_PyramidDepths,
																  vector<device_ptr<float4>>& d_PyramidVerts,
																  vector<device_ptr<float4>>& d_PyramidNormals)
{
	//Blur kernel runs top-down, i.e process higher resolutions first and go decreasing order.
	//[1] First raw depth
  int inWidth  = -1, outWidth  = -1;
	int inHeight = -1, outHeight = -1;

	int inPyrLvl  = -1;
	int outPyrLvl = -1;
	//for(int pyrLevel = pyramid_size-1; pyrLevel > 0; --pyrLevel)
	for(int pyrLevel = 0; pyrLevel < pyramid_size; ++pyrLevel)
	{
    inPyrLvl                    = pyrLevel;
    outPyrLvl                   = pyrLevel + 1;

		inWidth 												= pyramid_resolution[inPyrLvl][0] ;
		inHeight 												= pyramid_resolution[inPyrLvl][1] ;
		float4* d_referenceVertexMap    = thrust::raw_pointer_cast(d_PyramidVerts[0]); //inPyrLvl
		float4* d_referenceNormalMap    = thrust::raw_pointer_cast(d_PyramidNormals[0]); //inPyrLvl

		outWidth 												= pyramid_resolution[outPyrLvl][0] ;
		outHeight 											= pyramid_resolution[outPyrLvl][1] ;
		float4* d_toFillVertexMap 	  	= thrust::raw_pointer_cast(d_PyramidVerts[outPyrLvl]);
		float4* d_toFillNormalMap 	  	= thrust::raw_pointer_cast(d_PyramidNormals[outPyrLvl]);

    std::cout << "Downscaling depth frame (" << inWidth << ", " << inHeight << ") --> ("
																			 << outWidth << ", " << outHeight << ")"<<std::endl;

		int offset = pow(2, outPyrLvl);
		downscaleKernel<<<blocks[outPyrLvl], threads[outPyrLvl]>>>(d_referenceVertexMap, d_toFillVertexMap, inWidth, inHeight, outWidth, outHeight, offset);
		downscaleKernel<<<blocks[outPyrLvl], threads[outPyrLvl]>>>(d_referenceNormalMap, d_toFillNormalMap, inWidth, inHeight, outWidth, outHeight, offset);
		checkCudaErrors(cudaDeviceSynchronize());
	}

	// -- Update : not explicitly generating vertices for now. The solver doesn't seem to be converging with this ---
	/*
	//[2] and reconstruct camera-space vertices and normals
	//for(int pyrLevel = pyramid_size-1; pyrLevel > 0; --pyrLevel)
	for(int pyrLevel = 1; pyrLevel < pyramid_size; ++pyrLevel)
	{

		inWidth 											  = pyramid_resolution[pyrLevel][0] ;
		inHeight 											  = pyramid_resolution[pyrLevel][1] ;

		uint16_t* d_referenceDepthMap   = thrust::raw_pointer_cast(d_PyramidDepths[pyrLevel]);
		float4* d_toFillVertMap					= thrust::raw_pointer_cast(d_PyramidVerts[pyrLevel]);
		float4* d_toFillNormalMap				= thrust::raw_pointer_cast(d_PyramidNormals[pyrLevel]);

    std::cout << "Calculating Vertex on downscaled frame (" << inWidth << ", " << inHeight
			<<")" <<std::endl;

		calculateVertexPositions <<<blocks[pyrLevel], threads[pyrLevel] >>>(d_toFillVertMap, d_referenceDepthMap);
		checkCudaErrors(cudaDeviceSynchronize());

    std::cout << "Calculating Normals on downscaled frame (" << inWidth << ", " << inHeight
			<<")" <<std::endl;
		calculateNormals <<<blocks[pyrLevel], threads[pyrLevel] >>>(d_toFillVertMap, d_toFillNormalMap);
		checkCudaErrors(cudaDeviceSynchronize());
	}
	*/

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
