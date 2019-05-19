#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "cuda_helper/helper_math.h"
#include "cuda_helper/helper_cuda.h"
#include "VoxelDataStructures.h"

__constant__ HashTableParams d_hashtableParams;

VoxelEntry *d_hashTable;
VoxelEntry *d_compactifiedHashTable;
unsigned int *d_compactifiedHashCounter;
unsigned int *d_arena;	//Arena that manages free memory
unsigned int *d_arenaCounter;	//single element; points to next free block (atomic counter)
Voxel *d_voxelBlocks;
int *d_hashTableBucketMutex;	//mutex for locking particular bin while inserting/deleting


void updateConstantHashTableParams(const HashTableParams &params)	{
	size_t size;
	checkCudaErrors(cudaGetSymbolSize(&size, d_hashtableParams));
	checkCudaErrors(cudaMemcpyToSymbol(d_hashtableParams, &params, size, 0, cudaMemcpyHostToDevice));
}

__host__
void allocate(const HashTableParams &params)	{
	checkCudaErrors(cudaMalloc(&d_hashTable, sizeof(VoxelEntry) * params.numBuckets * params.bucketSize));
	checkCudaErrors(cudaMalloc(&d_arena, sizeof(unsigned int) * params.numVoxelBlocks));
	checkCudaErrors(cudaMalloc(&d_arenaCounter, sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc(&d_compactifiedHashTable, sizeof(VoxelEntry) * params.numBuckets * params.bucketSize));
	checkCudaErrors(cudaMalloc(&d_voxelBlocks, sizeof(unsigned int) * params.numVoxelBlocks * params.voxelBlockSize * params.voxelBlockSize * params.voxelBlockSize));
	checkCudaErrors(cudaMalloc(&d_hashTableBucketMutex, sizeof(int) * params.numBuckets));
}

__host__
void free()	{
	checkCudaErrors(cudaFree(d_hashTable));
	checkCudaErrors(cudaFree(d_arena));
	checkCudaErrors(cudaFree(d_arenaCounter));
	checkCudaErrors(cudaFree(d_compactifiedHashTable));
	checkCudaErrors(cudaFree(d_compactifiedHashCounter));
	checkCudaErrors(cudaFree(d_voxelBlocks));
	checkCudaErrors(cudaFree(d_hashTableBucketMutex));
}


//Now actual GPU code
__device__
unsigned int calculateHash(const int3& pos)	{
		const int p0 = 73856093;
		const int p1 = 19349669;
		const int p2 = 83492791;

		int res = ((pos.x * p0) ^ (pos.y * p1) ^ (pos.z * p2)) % d_hashtableParams.numBuckets;
		if (res < 0) res += d_hashtableParams.numBuckets;
		return (uint)res;
}

__device__
float getTruncation(float z)	{
	return d_hashtableParams.truncation + (d_hashtableParams.truncScale * z);
}

__device__
int3 voxel2Block(int3 voxel) 	{
	const int size = d_hashtableParams.voxelBlockSize;
	if(voxel.x < 0) voxel.x -= size-1;	//i.e voxelBlockSize -1
	if(voxel.y < 0) voxel.y -= size-1;
	if(voxel.z < 0) voxel.z -= size-1;
	return make_int3(voxel.x/size, voxel.y/size, voxel.z/size);
}

__device__
int3 world2Voxel(const float3& point)	{
	const int size = d_hashtableParams.voxelBlockSize;
	float3 p = point/size;
	return make_int3(p + make_int3(signbit(p))*0.5);//return center
}

__device__
int3 block2Voxel(const int3& block)	{
	return block*d_hashtableParams.voxelBlockSize;
}

__device__
float3 voxel2World(const int3& voxel)	{
	return make_float3(voxel) * d_hashtableParams.voxelSize;
}

__device__
float3 block2World(const int3& block)	{
	return voxel2World(block2Voxel(block));
}

__device__
int3 world2Block(const float3& point)	{
	return voxel2Block(world2Voxel(point));
}

__device__
unsigned int linearizeVoxelPos(const int3& pos)	{
	const int size = d_hashtableParams.voxelBlockSize;
	return  pos.z * size * size +
			pos.y * size +
			pos.x;
}

__device__
int3 delinearizeVoxelPos(const unsigned int& index)	{
	const int size = d_hashtableParams.voxelBlockSize;
	unsigned int x = index % size;
	unsigned int y = (index % (size * size)) / size;
	unsigned int z = index / (size * size);
	return make_int3(x,y,z);
}
