//Defines data structures for voxel, hashtable-entry, hashtable-params, voxel-block
#ifndef VOXEL_DATASTRUCTURES_H
#define VOXEL_DATASTRUCTURES_H

#include<cuda.h>
#include<cuda_runtime_api.h>
#include<cuda_runtime.h>
#include "cuda_helper/helper_math.h"
#include "cuda_helper/cuda_SimpleMatrixUtil.h"

__align__(8)
struct Voxel	{
	float sdf;
	//unsigned char weight;
	//unsigned int weight;
	float weight;
};

__align__(16)
struct VoxelEntry	{
	int3 pos;
	int ptr;
	int offset;
	//long padding1;
	//int padding2;
};

__align__(16)
struct HashTableParams	{
	//HashTableParams()	{
	//}

	float4x4 global_transform;
	float4x4 inv_global_transform;

	unsigned int numBuckets;	//= 500000;
	unsigned int bucketSize;	// = 10;
	unsigned int attachedLinkedListSize;	// = 7;
	unsigned int numVoxelBlocks;	// =1000000;

	int voxelBlockSize;	// = 8;
	float voxelSize;	// = 0.05f;
	unsigned int numOccupiedBlocks;	// = 0;	//occupied blocks in view frustum

	float maxIntegrationDistance;	// = 4.0f;
	float truncScale;	// = 0.01f;
	float truncation;	// = 0.02f;

	unsigned int integrationWeightSample;	// = 10;
	float integrationWeightMax;	// = 255;

};

struct PtrContainer	{
	unsigned int* d_heap;	//linear buffer with indices to all unallocated blocks
	VoxelEntry* d_hashTable;	//actual hashtable
	VoxelEntry* d_compactifiedHashTable;
	int* d_hashTableBucketMutex;	//mutex that'll decide wheter further allocations in bucket can be made or not
	Voxel* d_SDFBlocks;	//Actual underlying 3D geometry in TSDF form. Very important.

	int* d_heapCounter;	//single int keeping track of numBlocks allocated
	int* d_compactifiedHashCounter;
};

struct debug_ssbo_struct {
	unsigned int startPtr;
	float3 rayStartPos; //where ray hits the block in world-space
	unsigned int stopPtr;
	float3 rayStopPos; //where ray exits the block in world-space
};

#endif
