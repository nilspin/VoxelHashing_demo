#ifndef SDF_HASHTABLE_H
#define SDF_HASHTABLE_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_allocator.h>
#include <glm/glm.hpp>

#include "VoxelDataStructures.h"
#include "cuda_helper/cuda_SimpleMatrixUtil.h"
#include "cuda_helper/helper_cuda.h"

using namespace glm;
using namespace thrust;

extern HashTableParams d_hashtableParams;

class SDFRenderer;

class SDF_Hashtable {

private:
	HashTableParams h_hashtableParams;
	struct cudaGraphicsResource *numVisibleBlocks_res;
	struct cudaGraphicsResource *compactHashtable_res;
	struct cudaGraphicsResource *sdfBlocks_res;

public:

	SDF_Hashtable();
	~SDF_Hashtable();
	//Our hashtable needs to support these operations at the very least
	//void deleteHashEntry(uint id);
	//HashEntry deleteHashEntry(HashEntry& hashEntry);
	void integrate(const float4x4& deltaT, const float4*, const float4*);
	void registerGLtoCUDA(SDFRenderer&);
};

#endif
