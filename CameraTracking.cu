#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#include "CameraTracking.h"

using namespace glm;
#define MINF __int_as_float(0xff800000)
//Kinect v2 specific camera params
#define numCols 640
#define numRows 480
#define fx 525
#define fy 525
#define cx 319.5
#define cy 239.5

//Since numCols = 640 and numRows = 480, we set blockDim according to 32x32 tile
dim3 blocks = dim3(20,15,1);
dim3 threads = dim3(32,32,1);

__global__
void calculateVertexPositions(vec4* d_vertexPositions, const uint16_t* d_depthBuffer)  {
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
  vec4 vertex = vec4(x, -y, -depth, w);
  //*
  //if(idx<20){printf("thread: %d - %f %f %f %f\n", idx, vertex.x, vertex.y, vertex.z, vertex.w);}
  //*/
  d_vertexPositions[idx] = vertex;
}

__global__
void calculateNormals(const vec4* d_positions, vec4* d_normals)
{
  int xidx = blockDim.x*blockIdx.x + threadIdx.x;
  int yidx = blockDim.y*blockIdx.y + threadIdx.y;
  
  if(xidx >= numCols || yidx>= numRows) {
    return;
  }

  //find globalIdx row-major
  const int idx = (yidx*numCols)+xidx;

  d_normals[idx] = vec4(MINF, MINF, MINF, MINF);

  if(xidx > 0 && xidx < numCols-1 && yidx > 0 && yidx < numRows-1)  {
    const vec4 CC = d_positions[(yidx+0)*numCols+(xidx+0)];
		const vec4 PC = d_positions[(yidx+1)*numCols+(xidx+0)];
		const vec4 CP = d_positions[(yidx+0)*numCols+(xidx+1)];
		const vec4 MC = d_positions[(yidx-1)*numCols+(xidx+0)];
		const vec4 CM = d_positions[(yidx+0)*numCols+(xidx-1)];

    if(CC.x != MINF && PC.x != MINF && CP.x != MINF && MC.x != MINF && CM.x != MINF)
		{
			const vec3 n = cross(vec3(PC)-vec3(MC), vec3(CP)-vec3(CM));
			const float  l = length(n);

			if(l > 0.0f)
			{
        //float4 v = make_float4(n/-l, 1.0f);
        vec4 vert = vec4(n, l);
				d_normals[yidx*numCols+xidx] = vert;
        //printf("Normal for thread %d : %f %f %f", yidx*numRows+xidx, vert.x, vert.y, vert.z);
			}
		}
  }
  
}

void CameraTracking::Align(vec4* d_input, vec4* d_inputNormals, vec4* d_target, 
  vec4* d_targetNormals, const uint16_t* d_depthInput, const uint16_t* d_depthTarget) {

  preProcess(d_input, d_inputNormals, d_depthInput);
  preProcess(d_target, d_targetNormals, d_depthTarget);

}

//Takes device pointers, calculates correct position and normals
void CameraTracking::preProcess(vec4 *positions, vec4* normals, const uint16_t *depth)  {
  calculateVertexPositions<<<blocks, threads>>>(positions, depth);
  calculateNormals<<<blocks, threads>>>(positions, normals);
}