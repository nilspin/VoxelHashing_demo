#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "cuda_helper/helper_math.h"
#include "cuda_helper/helper_cuda.h"
#include "VoxelDataStructures.h"
#include "common.h"

//This is a simple vector math library. Use this with CUDA instead of glm
#include "cuda_helper/cuda_SimpleMatrixUtil.h"

#define FREE_BLOCK -1
#define LOCKED_BLOCK -2
#define NO_OFFSET 0

__constant__ HashTableParams d_hashtableParams;
__device__ __constant__ float4x4 kinectProjectionMatrix;

VoxelEntry *d_hashTable;
VoxelEntry *d_compactifiedHashTable;
unsigned int *d_compactifiedHashCounter;
unsigned int *d_arena;	//Arena that manages free memory
unsigned int *d_arenaCounter;	//single element; points to next free block (atomic counter)
Voxel *d_voxelBlocks;
int *d_hashTableBucketMutex;	//mutex for locking particular bin while inserting/deleting

__host__
void calculateKinectProjectionMatrix()	{
	float m[4][4];
	m[0][0] = 2.0 * fx / imgWidth;
    m[0][1] = 0.0;
    m[0][2] = 0.0;
    m[0][3] = 0.0;

    m[1][0] = 0.0;
    m[1][1] = -2.0 * fy / imgHeight;
    m[1][2] = 0.0;
    m[1][3] = 0.0;

    m[2][0] = 1.0 - 2.0 * cx / imgWidth;
    m[2][1] = 2.0 * cy / imgHeight - 1.0;
    m[2][2] = (kinZFar + kinZNear) / (kinZNear - kinZFar);
    m[2][3] = -1.0;

    m[3][0] = 0.0;
    m[3][1] = 0.0;
    m[3][2] = 2.0 * kinZFar * kinZNear / (kinZNear - kinZFar);
    m[3][3] = 0.0;

	//Now upload to device
	cudaCheckErrors(cudaMemcpyToSymbol(kinectProjectionMatrix, m, sizeof(m)));
}


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


//what follows is IMO coolest code in project so far
__device__
VoxelEntry getVoxelEntry4Block(const int3& pos)	{
	const unsigned int hash = calculateHash(pos);
	const unsigned int bucketSize = d_hashtableParams.bucketSize;
	const unsigned int numBuckets = d_hashtableParams.numBuckets;
	const unsigned int startIndex = hash * bucketSize;

	VoxelEntry temp;
	temp.pos = pos;
	temp.offset = 0;
	temp.ptr = FREE_BLOCK;

	int i=0;
	//[1] Iterate all bucketSize entries
	for(i=0; i < bucketSize ; ++i)	{
		VoxelEntry& curr = d_hashTable[startIndex + i];
		if((curr.pos.x == pos.x) && (curr.pos.y == pos.y) &&(curr.pos.z == pos.z)
				&& (curr.ptr != FREE_BLOCK)) {
			return curr;
		}
	}
	//[2] block not found. handle collisions by traversing tail linked list
	const int lastEntryInBucket = (hash+1)*bucketSize -1;
	i = lastEntryInBucket;
	//memorize idx at list end and memorize offset from last
	//element of bucket to list end
	int iter = 0;
	const int maxIter = d_hashtableParams.attachedLinkedListSize;
	while(iter < maxIter)	{

		VoxelEntry curr = d_hashTable[i];
		if((curr.pos.x == pos.x) && (curr.pos.y == pos.y) &&(curr.pos.z == pos.z)
				&& (curr.ptr != FREE_BLOCK)) {
			return curr;
		}

		if(curr.offset == 0)	{ //we've found end of list
			break;
		}
		i = lastEntryInBucket + curr.offset;

		i %= (numBuckets * bucketSize);

		iter++;
	}
	return temp;
}


__device__
bool insertVoxelEntry(const int3& pos)	{

	unsigned int hash = calculateHash(pos);
	const unsigned int bucketSize = d_hashtableParams.bucketSize;
	const unsigned int numBuckets = d_hashtableParams.numBuckets;
	const unsigned int startIndex = hash * bucketSize;

	VoxelEntry temp;
	temp.offset=0;
	temp.ptr = FREE_BLOCK;
	temp.pos = pos;

	//[1] iterate current bucket, try inserting at first empty block we see.
	int i=0;
	for(i=0; i<bucketSize; ++i)	{
		const int idx = startIndex+i;
		VoxelEntry &curr = d_hashTable[idx];
		if(curr.pos.x == pos.x && curr.pos.y == pos.y && curr.pos.z == pos.z
				&& curr.ptr != FREE_BLOCK)	return false;
		if(curr.ptr == FREE_BLOCK)	{
			int prevVal = atomicExch(&d_hashTableBucketMutex[idx], LOCKED_BLOCK);
			if(prevVal != LOCKED_BLOCK)	{	//means we can lock current bucket
				VoxelEntry &entry = d_hashTable[idx];
				entry.pos = pos;
				entry.offset = NO_OFFSET;
				entry.ptr = allocSingleBlockInHeap();
				return true;
			}
		}
	}
	//[2] bucket is full. Append to list.
	const int lastEntryInBucket = (hash+1)*bucketSize - 1;

	i = lastEntryInBucket;
	int offset=0;
	//memorize idx at list end and memorize offset from last
	//element of bucket to list end
	int iter = 0;
	const int maxIter = d_hashtableParams.attachedLinkedListSize;
	while(iter < maxIter)	{
		offset++;
		i = (lastEntryInBucket + offset)%(numBuckets*bucketSize);
		VoxelEntry &curr = d_hashTable[i];
		//if(curr.offset == 0)	continue;	//cannot insert in last bucket element
		if(curr.pos.x == pos.x && curr.pos.y == pos.y && curr.pos.z == pos.z
				&& curr.ptr != FREE_BLOCK)	return false;	//block already exists
		if(curr.ptr == FREE_BLOCK)	{	//first free entry
			//lock parent bucket
			int prevVal = atomicExch(&d_hashTableBucketMutex[hash], LOCKED_BLOCK);
			if(prevVal != LOCKED_BLOCK)	{
				hash = i/bucketSize;
				//lock bucket where free entry actually exists
				prevVal = atomicExch(&d_hashTableBucketMutex[hash], LOCKED_BLOCK);
				if(prevVal != LOCKED_BLOCK)	{
					curr.pos = pos;
					curr.offset = d_hashTable[lastEntryInBucket].offset;
					curr.ptr = allocSingleBlockInHeap();
					d_hashTable[lastEntryInBucket].offset = offset;
				}
}
