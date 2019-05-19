//Defines data structures for voxel, hashtable-entry, hashtable-params, voxel-block

#include<cuda.h>
#include<cuda_runtime_api.h>
#include<cuda_runtime.h>
#include "cuda_helper/helper_math.h"
#include "cuda_helper/cuda_SimpleMatrixUtil.h"

__align__(8)
struct Voxel	{
	float sdf;
	unsigned char weight;
};

__align__(16)
struct VoxelEntry	{
	int3 pos;
	unsigned int ptr;
	unsigned int offset;
};

__align__(16)
struct HashTableParams	{
	HashTableParams()	{
	}

	float4x4 global_transform;
	float4x4 inv_global_transform;

	unsigned int numBuckets;
	unsigned int bucketSize;
	unsigned int attachedLinkedListSize;
	unsigned int numVoxelBlocks;

	int voxelBlockSize;
	float voxelSize;
	unsigned int numOccupiedBlocks;	//occupied blocks in view frustum

	float maxIntegrationDistance;
	float truncScale;
	float truncation;

	unsigned int integrationWeightSample;
	unsigned int integrationWeightMax;

};



