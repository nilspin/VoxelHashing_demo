#ifndef SDF_HASHTABLE_H
#define SDF_HASHTABLE_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_allocator.h>
#include <glm/glm.hpp>

#include "VoxelDataStructures.h"
#include "cuda_helper/cuda_SimpleMatrixUtil.h"

using namespace glm;
using namespace thrust;

extern HashTableParams d_hashtableParams;

class SDF_Hashtable {

public:

	SDF_Hashtable();
	~SDF_Hashtable();
	//Our hashtable needs to support these operations at the very least
	//void deleteHashEntry(uint id);
	//HashEntry deleteHashEntry(HashEntry& hashEntry);
	void integrate(const float4x4& deltaT, const float4*, const float4*);
};

#endif
