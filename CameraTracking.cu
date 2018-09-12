#include<cuda_runtime.h>
#include<cuda_runtime_api.h>
#include <glm/glm.hpp>

#include "CameraTracking.h"

using namespace glm;

//Since numCols = 640 and numRows = 480, we set blockDim according to 32x32 tile
dim3 blocks = dim3(20,15,1);
dim3 threads = dim3(32,32,1);

//Kinect v2 specific camera params
#define numCols 640
#define numRows 480
#define fx 525
#define fy 525
#define cx 319.5
#define cy 239.5

__global__
void calculateVertexPositions(vec4* d_vertexPositions, uint16_t* d_depthBuffer)  {
  int xidx = blockDim.x*blockIdx.x + threadIdx.x;
  int yidx = blockDim.y*blockIdx.y + threadIdx.y;
  
  if(xidx >= numCols || yidx>= numRows) {
    return;
  }

  const int idx = yidx*numCols + xidx;

  float w = 0.0f; //flag to tell whether this is valid vertex or not
  float depth = d_depthBuffer[idx];
  if(depth != 0) {w = 1.0f;}
  float x = ((xidx - cx)*depth)/fx;
  float y = ((yidx - cy)*depth)/fy;
  vec4 vertex = vec4(x, y, depth, w);
  d_vertexPositions[idx] = vertex;
}

__global__
void calculateNormals(vec4* d_positions, vec4* d_normals)
{
  int xidx = blockDim.x*blockIdx.x + threadIdx.x;
  int yidx = blockDim.y*blockIdx.y + threadIdx.y;
  
  if(xidx >= numCols || yidx>= numRows) {
    return;
  }

  
}

void CameraTracking::Align(vec4* d_input, vec4* d_inputNormals, vec4* d_target, 
  vec4* d_targetNormals, uint16_t* d_depthInput, uint16_t* d_depthTarget) {

  preProcess(d_input, d_inputNormals, d_depthInput);
  preProcess(d_target, d_targetNormals, d_depthTarget);

}

//Takes device pointers, calculates correct position and normals
void CameraTracking::preProcess(vec4 *positions, vec4* normals, uint16_t *depth)  {
  calculateVertexPositions<<<blocks, threads>>>(positions, depth);
  calculateNormals<<<blocks, threads>>>(positions, normals);
}