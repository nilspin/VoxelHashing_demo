#ifndef CAMTRACKING_UTIL
#define CAMTRACKING_UTIL

#include <Windows.h>

#include <cuda_runtime_api.h>
#include "cuda_helper/helper_cuda.h"
#include "cuda_helper/helper_math.h"

//This is a simple vector library. Use this with CUDA instead of GLM.
#include "cuda_helper/cuda_SimpleMatrixUtil.h"

#define MINF __int_as_float(0xff800000)
#define MAXF __int_as_float(0x7F7FFFFF)
#define fx 525
#define fy 525
#define cx 319.5
#define cy 239.5
//Kinect v2 specific camera params
#define numCols 640
#define numRows 480

const float distThres = 2.0;
const float normalThres = -1.f;
//Since numCols = 640 and numRows = 480, we set blockDim according to 32x32 tile
dim3 blocks = dim3(20, 15, 1);
dim3 threads = dim3(32, 32, 1);


__device__
static inline int2 cam2screenPos(float3 &p) {
	float x = ((p.x * fx) / p.z) + cx;
	float y = ((p.y * fy) / p.z) + cy;
	return make_int2(x, y);
}

__global__
void calculateVertexPositions(float4* d_vertexPositions, const uint16_t* d_depthBuffer) {
	int xidx = blockDim.x*blockIdx.x + threadIdx.x;
	int yidx = blockDim.y*blockIdx.y + threadIdx.y;

	if (xidx >= numCols || yidx >= numRows) {
		return;
	}

	//find globalIdx row-major
	const int idx = (yidx*numCols) + xidx;

	float w = 1.0f; //flag to tell whether this is valid vertex or not
	uint16_t d = d_depthBuffer[idx];
	float depth = d / 5000.0f; //5000 units = 1meter. We're now dealing in meters.
	if (depth == 0) {
		w = 0.0f;
	}

	float x = ((xidx - cx)*depth) / (float)fx;
	float y = ((yidx - cy)*depth) / (float)fy;
	float4 vertex = make_float4(x, -y, -depth, w);
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

	d_normals[idx] = make_float4(MINF, MINF, MINF, MINF);

	if (xidx > 0 && xidx < numCols - 1 && yidx > 0 && yidx < numRows - 1) {
		const float4 CC = d_positions[(yidx + 0)*numCols + (xidx + 0)];
		const float4 PC = d_positions[(yidx + 1)*numCols + (xidx + 0)];
		const float4 CP = d_positions[(yidx + 0)*numCols + (xidx + 1)];
		const float4 MC = d_positions[(yidx - 1)*numCols + (xidx + 0)];
		const float4 CM = d_positions[(yidx + 0)*numCols + (xidx - 1)];

		if (CC.x != MINF && PC.x != MINF && CP.x != MINF && MC.x != MINF && CM.x != MINF)
		{
			const float3 n = cross(make_float3(PC) - make_float3(MC), make_float3(CP) - make_float3(CM));
			const float  l = length(n);

			if (l > 0.0f)
			{
				//float4 v = make_float4(n/-l, 1.0f);
				float4 vert = make_float4(n, l);
				d_normals[yidx*numCols + xidx] = vert;
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

__global__
void FindCorrespondences(const float4* input, const float4* inputNormals,
	const float4* target, const float4* targetnormals,
	float4* correspondence, float4* correspondenceNormals,
	const float4x4 deltaT, float distThres, float normalThres, int width, int height) {

	int xidx = blockDim.x*blockIdx.x + threadIdx.x;
	int yidx = blockDim.y*blockIdx.y + threadIdx.y;

	if (xidx >= numCols || yidx >= numRows) {
		return;
	}

	const int idx = (yidx*numCols) + xidx;

	correspondence[idx] = make_float4(MINF, MINF, MINF, MINF);
	correspondenceNormals[idx] = make_float4(MINF, MINF, MINF, MINF);
	float4 p_in = input[idx];
	float4 n_in = inputNormals[idx];

	if (p_in.w != 0 && n_in.w != 0) {	//if both pos and normal are valid points

		float4 transformedPos = deltaT*p_in;
		float4 transformedNormal = deltaT*n_in;

		int2 screenPos = cam2screenPos(make_float3(transformedPos));
//		if (idx == 25214) {
//			printf("translated pos:(%f, %f, %f) screenPos = ( %d, %d ) for thread %d blockX %d blockY %d \n", p.x, p.y, p.z, screenPos.x, screenPos.y, idx, blockIdx.x, blockIdx.y);
//		}

		//now lookup this index in target image
		float4 p_target, n_target;
		int linearScreenPos = (screenPos.y*numCols) + screenPos.x;

		if (screenPos.x >= numCols || screenPos.x < 0 || screenPos.y >= numRows || screenPos.y < 0) {
			return;
		}

		p_target = target[linearScreenPos];
		n_target = targetnormals[linearScreenPos];
		
		if (p_target.w != 0.0f) {
			float n = dot(make_float3(transformedNormal), make_float3(n_target));
			float d = length(make_float3(transformedPos) - make_float3(p_target));
			if (d <= distThres && n >= normalThres) {
				correspondence[idx] = p_target;
				correspondenceNormals[idx] = n_target;
			}
		}
	}
}

extern "C" void computeCorrespondences(const float4* d_input, const float4* d_inputNormals, const float4* d_target, const float4* d_targetNormals, float4* d_correspondence, float4* d_correspondenceNormals,
	const float4x4 deltaTransform, const int width, const int height)
{
	//First clear the previous correspondence calculation
	const int ARRAY_SIZE = width*height * sizeof(float4);
	checkCudaErrors(cudaMemset(d_correspondence, 0, ARRAY_SIZE));
	checkCudaErrors(cudaMemset(d_correspondenceNormals, 0, ARRAY_SIZE));

	FindCorrespondences <<<blocks, threads >>>(d_input, d_inputNormals, d_target, d_targetNormals, d_correspondence, d_correspondenceNormals,
		deltaTransform, distThres, normalThres, width, height);
	checkCudaErrors(cudaDeviceSynchronize());

}

#endif // CAMTRACKING_UTIL