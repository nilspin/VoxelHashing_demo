#ifndef CAMERA_TRACKING_H
#define CAMERA_TRACKING_H

#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_CUDA
#define GLM_SWIZZLE

#include<Windows.h>
#include<glm/glm.hpp>
#include<cuda_runtime_api.h>
#include<cuda.h>
#include<cuda_gl_interop.h>
#include "cudaHelper.h"

using glm::vec3;
using glm::vec4;
using glm::mat4;

class CameraTracking  {

private:
  int width, height;
  
  vec4* d_correspondenceNormals;
  vec4* d_correspondence;
  mat4 deltaTransform;
  void preProcess(vec4 *, vec4*, const uint16_t*);
public:
  
  CameraTracking(int, int);
  ~CameraTracking();
  //void FindCorrespondences(const vec4*, const vec4*, const vec4*, const vec4*, vec4*, vec4*, const mat4&, int, int);
  void Align(vec4*, vec4*, vec4*, vec4*, const uint16_t*, const uint16_t*);
};

__global__
void FindCorrespondences(const vec4*, const vec4*, const vec4*, const vec4*, vec4*, vec4*, const mat4&, int, int);

__device__
static inline int2 cam2screenPos(const vec3&);

__global__
void calculateVertexPositions(vec4* , const uint16_t*);

__global__
void calculateNormals(const vec4* , vec4*);

#endif 