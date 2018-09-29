#include <Windows.h>

#include <helper_cuda.h>
#include <helper_math.h>
#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
//#include <glm/gtx/string_cast.hpp>
#include "CameraTracking.h"

//using namespace glm;
#define MINF __int_as_float(0xff800000)
//Kinect v2 specific camera params
#define numCols 640
#define numRows 480
#define fx 525
#define fy 525
#define cx 319.5
#define cy 239.5

const int distThreshold = 2.0;
//Since numCols = 640 and numRows = 480, we set blockDim according to 32x32 tile
dim3 blocks = dim3(20,15,1);
dim3 threads = dim3(32,32,1);


__device__
static inline int2 cam2screenPos(float3 p) {
  float x = ((p.x * fx)/p.z) + cx;
  float y = ((p.y * fy)/p.z) + cy;
  return make_int2(x,y);
}

__global__
void calculateVertexPositions(float4* d_vertexPositions, const uint16_t* d_depthBuffer)  {
  int xidx = blockDim.x*blockIdx.x + threadIdx.x;
  int yidx = blockDim.y*blockIdx.y + threadIdx.y;
  
  if(xidx >= numCols || yidx>= numRows) {
    return;
  }

  //find globalIdx row-major
  const int idx = (yidx*numCols)+xidx;


  /*Debug stuff
  if(threadIdx.x==0 && threadIdx.y==0)  {
    printf("blockIdx.x : %d , blockIdx.y : %d , blockNo : %d \n", blockIdx.x, blockIdx.y, blockIdx.y*gridDim.x + blockIdx.x);
  }*/
  
  float w = 1.0f; //flag to tell whether this is valid vertex or not
  uint16_t d = d_depthBuffer[idx];
  float depth = d/5000.0f; //5000 units = 1meter. We're now dealing in meters.
  if(depth == 0) {
    w = 0.0f;
  }
  //printf("%d : %f \n",idx, depth);
  
  float x = ((xidx - cx)*depth)/(float)fx;
  float y = ((yidx - cy)*depth)/(float)fy;
  float4 vertex = make_float4(x, -y, -depth, w);
  //*
  //if(idx<20){printf("thread: %d - %f %f %f %f\n", idx, vertex.x, vertex.y, vertex.z, vertex.w);}
  //*/
  d_vertexPositions[idx] = vertex;
}

__global__
void calculateNormals(const float4* d_positions, float4* d_normals)
{
  int xidx = blockDim.x*blockIdx.x + threadIdx.x;
  int yidx = blockDim.y*blockIdx.y + threadIdx.y;
  
  if(xidx >= numCols || yidx>= numRows) {
    return;
  }

  //find globalIdx row-major
  const int idx = (yidx*numCols)+xidx;

  d_normals[idx] = make_float4(MINF, MINF, MINF, MINF);

  if(xidx > 0 && xidx < numCols-1 && yidx > 0 && yidx < numRows-1)  {
    const float4 CC = d_positions[(yidx+0)*numCols+(xidx+0)];
		const float4 PC = d_positions[(yidx+1)*numCols+(xidx+0)];
		const float4 CP = d_positions[(yidx+0)*numCols+(xidx+1)];
		const float4 MC = d_positions[(yidx-1)*numCols+(xidx+0)];
		const float4 CM = d_positions[(yidx+0)*numCols+(xidx-1)];

    if(CC.x != MINF && PC.x != MINF && CP.x != MINF && MC.x != MINF && CM.x != MINF)
		{
			const float3 n = cross(make_float3(PC)-make_float3(MC), make_float3(CP)-make_float3(CM));
			const float  l = length(n);

			if(l > 0.0f)
			{
				//float4 v = make_float4(n/-l, 1.0f);
				float4 vert = make_float4(n, l);
				d_normals[yidx*numCols+xidx] = vert;
				//printf("Normal for thread %d : %f %f %f", yidx*numRows+xidx, vert.x, vert.y, vert.z);
			}
		}
  }
  
}

void CameraTracking::Align(float4* d_input, float4* d_inputNormals, float4* d_target, 
  float4* d_targetNormals, const uint16_t* d_depthInput, const uint16_t* d_depthTarget) {

  preProcess(d_input, d_inputNormals, d_depthInput);
  preProcess(d_target, d_targetNormals, d_depthTarget);

  //We now have all data we need. find correspondence.
  //glm::mat4 deltaT = mat4(1);
  checkCudaErrors(cudaDeviceSynchronize());
  
//  FindCorrespondences<<<blocks, threads>>>(d_input, d_inputNormals, d_target, d_targetNormals, d_correspondence, d_correspondenceNormals,
//                      deltaTransform, width, height);

  //checkCudaErrors(cudaDeviceSynchronize());
  // std::vector<glm::vec4> outputCUDA(640*480);
  // std::cout<<"outputCuda.size()"<<outputCUDA.size()<<std::endl;
  // checkCudaErrors(cudaMemcpy(outputCUDA.data(), d_correspondence, 640*480*sizeof(glm::vec4), cudaMemcpyDeviceToHost));
  // std::ofstream fout("correspondenceData.txt");
  // std::for_each(outputCUDA.begin(), outputCUDA.end(), [&fout](const glm::vec4 &n){fout<<n.x<<" "<<n.y<<" "<<n.z<<" "<<n.w<<"\n";});
  // fout.close();
                     
}

//Takes device pointers, calculates correct position and normals
void CameraTracking::preProcess(float4 *positions, float4* normals, const uint16_t *depth)  {
  calculateVertexPositions<<<blocks, threads>>>(positions, depth);
  calculateNormals<<<blocks, threads>>>(positions, normals);
}

CameraTracking::CameraTracking(int w, int h):width(w),height(h)
{
  const int ARRAY_SIZE = width*height*sizeof(float4);
  checkCudaErrors(cudaMalloc((void**)&d_correspondence, ARRAY_SIZE));
  checkCudaErrors(cudaMemset(d_correspondence, 0, ARRAY_SIZE));
  checkCudaErrors(cudaMalloc((void**)&d_correspondenceNormals, ARRAY_SIZE));
  checkCudaErrors(cudaMemset(d_correspondenceNormals, 0, ARRAY_SIZE));
}

CameraTracking::~CameraTracking()
{
  checkCudaErrors(cudaFree(d_correspondence));
  checkCudaErrors(cudaFree(d_correspondenceNormals));
}

__global__
void FindCorrespondences(const float4* input, const float4* inputNormals, 
                                        const float4* target, const float4* targetnormals, 
                                        float4* correspondence, float4* correspondenceNormals,
const float4x4 deltaT, int width, int height) {

  int xidx = blockDim.x*blockIdx.x + threadIdx.x;
  int yidx = blockDim.y*blockIdx.y + threadIdx.y;
  
  if(xidx >= numCols || yidx>= numRows) {
    return;
  }

  //find globalIdx row-major
  const int idx = (yidx*numCols)+xidx;

  float4 p_in = input[idx];
  float4 p = p_in;
  //vec4 translated = vec4(0,0,0,0);
  //vec4 translated = deltaT*p_in;

  
  //At what index does translated input lie?
  //vec3 p = vec3(translated.x,translated.y,translated.z);
  //float x = ((p.x * 525)/p.z) + 319.5;
  //float y = ((p.y * 525)/p.z) + 239.5;

  //int2 screenPos = make_int2(x,y);
  //printf("screenPos = ( %f, %f ) for thread %d \n", x, y, idx);
  
  /*
  //now lookup this index in target image
  vec4 p_target;//, n_target;
  int linearScreenPos = (screenPos.y*numCols)+screenPos.x;

  if(screenPos.x >= numCols || screenPos.y>= numRows) {
    return;
  }

  p_target = target[linearScreenPos];
  if(p_target.w == 1.0f)  {
    float d = length(translated - p_target);
    if(d < distThreshold) {
      correspondence[idx] = p_target;
      printf("thread %d, correspondence : %f %f %f %f \n", idx, p_target.x, p_target.y, p_target.z, p_target.w);
      //Don't need normal of correspondence right now, 
      //but many papers seem to use this for weighted ICP.
      //Might need later so set to 0.
      correspondenceNormals[idx] = vec4(0);
    }
  }
  */
}

