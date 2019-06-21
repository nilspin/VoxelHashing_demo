#ifdef _WIN32
#include <windows.h>
#endif

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include "VoxelDataStructures.h"
#include "VoxelUtils.h"
#include "common.h"

//This is a simple vector math library. Use this with CUDA instead of glm
#include "cuda_helper/cuda_SimpleMatrixUtil.h"

#define FREE_BLOCK -1
#define LOCKED_BLOCK -2
#define NO_OFFSET 0

__constant__ HashTableParams d_hashtableParams;
__constant__ float3x3 kinectProjectionMatrix;
__device__ PtrContainer ptrHldr;


__inline__ __device__
bool FIRST_THREAD()	{
	if(threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0)	{
		return true;
	}
	return false;
}

__inline__ __device__
void printDeviceMatrix(const float3x3& mat)	{

	printf("Printing device matrix...\n");
	for(int i=0;i<3;++i)	{
		for(int j=0;j<3;++j)	{
			printf("%f\t", mat.entries2[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}
/*----------(Raw)STORAGE---------------*/
//VoxelEntry *d_hashTable;
//VoxelEntry *d_compactifiedHashTable;
//int* d_compactifiedHashCounter;
//unsigned int *d_compactifiedHashCounter;
//int *d_heap;	//Arena that manages free memory
//int *d_heapCounter;	//single element; points to next free block (atomic counter)
//__device__ int d_heapCounter;
//Voxel *d_voxelBlocks;
//int *d_hashTableBucketMutex;	//mutex for locking particular bin while inserting/deleting
/*------------------------------------*/

/*---------(Thrust)STORAGE--------------*/

//hashtable
//thrust::device_vector<VoxelEntry> d_hashTable_vec;
//thrust::device_vector<VoxelEntry> d_compactifiedHashTable_vec;
//thrust::device_vector<int> d_hashTableBucketMutex_vec;
//__device__ int d_compactifiedHashCounter;

//VoxelEntry* d_hashTable;
//VoxelEntry* d_compactifiedHashTable;
//int* d_hashTableBucketMutex;

//heap management
//thrust::device_vector<int> d_heap_vec;	//heap that manages free memory
//__device__ int d_heapCounter;

//int* d_heap;

//actual voxelblocks
//thrust::device_vector<Voxel> d_SDFBlocks_vec;	//main heap holding tsdf blocks

//Voxel* d_SDFBlocks;
/*------------------------------*/

//! Make rigid transform available on the device
void updateConstantHashTableParams(const HashTableParams &params)	{
	size_t size;
	checkCudaErrors(cudaGetSymbolSize(&size, d_hashtableParams));
	checkCudaErrors(cudaMemcpyToSymbol(d_hashtableParams, &params, size, 0, cudaMemcpyHostToDevice));
}

//__host__
//void allocate(const HashTableParams& params)	{
//	const int initVal = 0;
//	d_hashTable_vec.resize(params.numBuckets * params.bucketSize);
//	d_compactifiedHashTable_vec.resize(params.numBuckets * params.bucketSize);
//	d_hashTableBucketMutex_vec.resize(params.numBuckets * params.bucketSize);
//	d_heap_vec.resize(params.numBuckets * params.bucketSize);
//	d_SDFBlocks_vec.resize(params.numBuckets * params.bucketSize * 512);
//	checkCudaErrors(cudaMemcpyToSymbol(ptrHldr.d_heapCounter, &initVal, sizeof(int),
//			0, cudaMemcpyHostToDevice));
//	checkCudaErrors(cudaMemcpyToSymbol(ptrHldr.d_compactifiedHashCounter, &initVal,
//			sizeof(int), 0, cudaMemcpyHostToDevice));
//	//init raw pointers
//	ptrHldr.d_heap = thrust::raw_pointer_cast(&d_heap_vec[0]);
//	ptrHldr.d_hashTable = thrust::raw_pointer_cast(&d_hashTable_vec[0]);
//	ptrHldr.d_compactifiedHashTable = thrust::raw_pointer_cast(&d_compactifiedHashTable_vec[0]);
//	ptrHldr.d_hashTableBucketMutex = thrust::raw_pointer_cast(&d_hashTableBucketMutex_vec[0]);
//	ptrHldr.d_SDFBlocks = thrust::raw_pointer_cast(&d_SDFBlocks_vec[0]);
//}

//TODO do this using thrust instead
__host__
void deviceAllocate(const HashTableParams &params)	{
	checkCudaErrors(cudaMalloc(&ptrHldr.d_heap, sizeof(unsigned int) * params.numVoxelBlocks));
	checkCudaErrors(cudaMalloc(&ptrHldr.d_hashTable, sizeof(VoxelEntry) * params.numBuckets * params.bucketSize));
	checkCudaErrors(cudaMalloc(&ptrHldr.d_compactifiedHashTable, sizeof(VoxelEntry) * params.numBuckets * params.bucketSize));
	checkCudaErrors(cudaMalloc(&ptrHldr.d_hashTableBucketMutex, sizeof(int) * params.numBuckets));
	checkCudaErrors(cudaMalloc(&ptrHldr.d_SDFBlocks, sizeof(Voxel) * params.numVoxelBlocks * params.voxelBlockSize * params.voxelBlockSize * params.voxelBlockSize));
	checkCudaErrors(cudaMalloc(&ptrHldr.d_heapCounter, sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc(&ptrHldr.d_compactifiedHashCounter, sizeof(int)));
}

__host__
void deviceFree()	{
	checkCudaErrors(cudaFree(ptrHldr.d_hashTable));
	checkCudaErrors(cudaFree(ptrHldr.d_heap));
	checkCudaErrors(cudaFree(ptrHldr.d_heapCounter));
	checkCudaErrors(cudaFree(ptrHldr.d_compactifiedHashTable));
	checkCudaErrors(cudaFree(ptrHldr.d_compactifiedHashCounter));
	checkCudaErrors(cudaFree(ptrHldr.d_SDFBlocks));
	checkCudaErrors(cudaFree(ptrHldr.d_hashTableBucketMutex));
}

__host__
void calculateKinectProjectionMatrix()	{

	float3x3 m(intrinsics);
	//Now upload to device
	std::cout<<"Uploading projection matrix to device..\n";
	checkCudaErrors(cudaMemcpyToSymbol(kinectProjectionMatrix, &m, sizeof(m)));
}

//TODO : Remove this function later
__inline__ __device__
bool vertexInFrustum(float4 point)	{
	point = d_hashtableParams.global_transform * point;
	float3 pos = make_float3(point.x, point.y, point.z);
	pos = kinectProjectionMatrix * pos;
	pos = pos/pos.z;	//normalize, and get screen coordinates
	int x = __float2int_rz(pos.x);
	int y = __float2int_rz(pos.y);
	if(x < 640 && x >=0 && y < 480 && y >= 0)	{
		return true;
	}
	return false;
}

__inline__ __device__
bool blockInFrustum(int3 blockId)	{
	float4 pos = make_float4(blockId.x, blockId.y, blockId.z, 1);
	pos = d_hashtableParams.global_transform * pos;
	float3 projected = make_float3(blockId.x, blockId.y, blockId.z);
	projected = kinectProjectionMatrix * projected;
	projected = projected/projected.z;

	int x = __float2int_rz(projected.x);
	int y = __float2int_rz(projected.y);
	if(x < 640 && x >=0 && y < 480 && y >= 0)	{
		return true;
	}
	return false;
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
	return make_int3(p + make_float3(signbit(p.x), signbit(p.y),signbit(p.z))*0.5);//return center
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

__device__
int allocSingleBlockInHeap()	{	//int ptr
	//int delIdx = ptr / 512;
	uint addr = atomicSub(&ptrHldr.d_heapCounter[0], 1);
	return ptrHldr.d_heap[addr];
}

__device__
void removeSingleBlockInHeap(int ptr)	{
	//int delIdx = ptr / 512;
	uint addr = atomicAdd(&ptrHldr.d_heapCounter[0], 1);
	ptrHldr.d_heap[addr + 1] = ptr;
}


//Hacky but cool code below
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
		VoxelEntry& curr = ptrHldr.d_hashTable[startIndex + i];
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

		VoxelEntry curr = ptrHldr.d_hashTable[i];
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


//TODO incomplete function
__inline__ __device__
bool insertVoxelEntry(const int3& data)	{

	unsigned int hash = calculateHash(data);
	const unsigned int bucketSize = d_hashtableParams.bucketSize;
	const unsigned int numBuckets = d_hashtableParams.numBuckets;
	const unsigned int startIndex = hash * bucketSize;

	VoxelEntry temp;
	temp.offset=0;
	temp.ptr = FREE_BLOCK;
	temp.pos = data;

	//[1] iterate current bucket, try inserting at first empty block we see.
	int i=0;
	if(FIRST_THREAD())	{
		printf("Insertion : before bucket iteration\n");
	}
	for(i=0; i<bucketSize; ++i)	{
		const int idx = startIndex+i;
		VoxelEntry &curr = ptrHldr.d_hashTable[idx];
		if(curr.pos.x == data.x && curr.pos.y == data.y && curr.pos.z == data.z
				&& curr.ptr != FREE_BLOCK)	return false;
		if(curr.ptr == FREE_BLOCK)	{
			//TODO shouldn't the following be [hash] instead of [idx] ?
			int prevVal = atomicExch(&ptrHldr.d_hashTableBucketMutex[hash], LOCKED_BLOCK);
			if(prevVal != LOCKED_BLOCK)	{	//means we can lock current bucket
				curr.pos = data;
				curr.offset = NO_OFFSET;
				curr.ptr = allocSingleBlockInHeap();
				return true;
			}
		}
	}
	if(FIRST_THREAD())	{
		printf("Insertion: Empty slot not found in native bucket. Appending list\n");
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
		i = i%(numBuckets*bucketSize);
		VoxelEntry& curr = ptrHldr.d_hashTable[i];
		if(curr.ptr != FREE_BLOCK)	{
			if(curr.pos.x == data.x && curr.pos.y == data.y &&
					curr.pos.z == data.z && curr.ptr != FREE_BLOCK)	{
				return false;	//alloc unsuccessful because block already there
			}
			if(curr.offset == 0)	{//end of list, lookahead till we find empty slot
				int j=1;
				//[1] lock the parent block
				int prevVal = atomicExch(&ptrHldr.d_hashTableBucketMutex[hash],
						LOCKED_BLOCK);
				if(prevVal != LOCKED_BLOCK)	{//if we got the lock
					//[2] then lookahead for empty block in new bucket
					while(j<10)	{
						if(ptrHldr.d_hashTable[i+j].ptr == FREE_BLOCK)	break;
						j++;
					}
					if(j==10)	{
						//we couldn't find empty space despite looking ahead 10 spaces
						return false;
					}
					//[3] now lock this new bucket and insert the block
					prevVal = atomicExch(&ptrHldr.d_hashTableBucketMutex[(i+j)/numBuckets],
							LOCKED_BLOCK);
					if(prevVal != LOCKED_BLOCK)	{
						VoxelEntry& next = ptrHldr.d_hashTable[i+j];
						//TODO maybe we can do away with this check
						if(next.ptr == FREE_BLOCK)	{
							next.ptr = allocSingleBlockInHeap();
							next.pos = data;
							curr.offset = j;
							break;
						}
						i++;
					}
				}
			}
			//TODO: implement this correctly
			if(curr.offset != 0)	{	//traversing nodes in linked list
				int j = i;
				while(j <= (i+curr.offset))	{
					if(ptrHldr.d_hashTable[j].ptr == FREE_BLOCK)	{
						//[a] free space found. first lock bucket with curr
						int prevVal = atomicExch(&ptrHldr.d_hashTableBucketMutex[hash/numBuckets], LOCKED_BLOCK);
						if(prevVal != LOCKED_BLOCK)	{
							//[b] then lock bucket with new space
							prevVal = atomicExch(&ptrHldr.d_hashTableBucketMutex[j/numBuckets], LOCKED_BLOCK);
							if(prevVal != LOCKED_BLOCK)	{
								VoxelEntry& ins = ptrHldr.d_hashTable[j];
								ins.offset = i + curr.offset - j;
								ins.ptr = allocSingleBlockInHeap();
								ins.pos = data;
								curr.offset = j - i;
								return true;
							}
						}
					}
					j++;
				}
				i += curr.offset;
			}
		}
		iter++;
	}
}


__device__
int beforeThis(int3 data)	{

	unsigned int hash = calculateHash(data);
	const unsigned int bucketSize = d_hashtableParams.bucketSize;
	const unsigned int numBuckets = d_hashtableParams.numBuckets;
	const unsigned int lastEntryInBucket = (hash+1) * bucketSize - 1;

	int iter = 0; const int maxIter = d_hashtableParams.attachedLinkedListSize;
	int i = lastEntryInBucket;
	if(ptrHldr.d_hashTable[lastEntryInBucket].offset != 0)	{
		while(iter < maxIter)	{
			const VoxelEntry& curr = ptrHldr.d_hashTable[i];
			const VoxelEntry& next = ptrHldr.d_hashTable[i + curr.offset];
			if((next.pos.x==data.x) && (next.pos.y==data.y) &&
					(next.pos.z==data.z))	{return i;}
			i += curr.offset;
			iter++;
		}
	}
	return -1;	//error; should not happen
}

__device__
bool deleteVoxelEntry(int3 data)	{
	//TODO : iterate over entire bucket
	unsigned int hash = calculateHash(data);
	const unsigned int bucketSize = d_hashtableParams.bucketSize;
	const unsigned int numBuckets = d_hashtableParams.numBuckets;
	const unsigned int startIndex = hash * bucketSize;

	VoxelEntry temp;
	temp.offset=0;
	temp.ptr = FREE_BLOCK;
	temp.pos = data;

	//[1] iterate current bucket, try inserting at first empty block we see.
	int i=0;
	for(i=0; i<bucketSize; ++i)	{
		const int idx = startIndex+i;
		VoxelEntry &curr = ptrHldr.d_hashTable[idx];
		if(curr.pos.x == data.x && curr.pos.y == data.y && curr.pos.z == data.z
				&& curr.ptr != FREE_BLOCK)	{return false;}
		if(curr.ptr == FREE_BLOCK)	{
			//TODO shouldn't the following be [hash] instead of [idx] ?
			//try locking current bucket
			int prevVal = atomicExch(&ptrHldr.d_hashTableBucketMutex[hash], LOCKED_BLOCK);
			if(prevVal != LOCKED_BLOCK)	{	//means we can lock current bucket
				curr.pos = make_int3(0);
				curr.offset = NO_OFFSET;
				removeSingleBlockInHeap(curr.ptr);
				curr.ptr = FREE_BLOCK;
				return true;
			}
		}
	}
	//deletion in linked list
	int lastEntry = beforeThis(data);
	if(lastEntry == -1)	{return false;}	//error
	VoxelEntry& prev = ptrHldr.d_hashTable[lastEntry];
	VoxelEntry& curr = ptrHldr.d_hashTable[lastEntry + prev.offset];
	//lock the bucket with curr
	int prevVal = atomicExch(&ptrHldr.d_hashTableBucketMutex[hash], LOCKED_BLOCK);
	if(prevVal!=LOCKED_BLOCK)	{	//lock acquired
		prevVal = atomicExch(&ptrHldr.d_hashTableBucketMutex[lastEntry / numBuckets],
				LOCKED_BLOCK);
		if(prevVal != LOCKED_BLOCK)	{
			//TODO FINISH THIS!!!
			prev.offset += curr.offset;
			curr.pos = make_int3(0);
			curr.offset = NO_OFFSET;
			removeSingleBlockInHeap(curr.ptr);
			curr.ptr = FREE_BLOCK;
		}
	}

	return false;//delete didn't happen :(

}

__global__
void allocBlocksKernel(const float4* verts, const float4* normals)	{	//Do we need normal data here?

	const float voxSize = d_hashtableParams.voxelSize;
	int xidx = blockDim.x*blockIdx.x + threadIdx.x;
	int yidx = blockDim.y*blockIdx.y + threadIdx.y;

	if (xidx >= numCols || yidx >= numRows) {
		return;
	}

	//find globalIdx row-major
	const int idx = (yidx*numCols) + xidx;

	float4 tempPos = verts[idx];
	if(tempPos.z == 0.0f) return;
	tempPos = d_hashtableParams.global_transform * tempPos;	//transform to global frame
	float3 projTemp = make_float3(tempPos.x, tempPos.y, tempPos.z);
	projTemp = kinectProjectionMatrix * projTemp;
	projTemp = projTemp/projTemp.z;
	//TODO : Erase this later
	//if(idx==153600)	{
	//	printf("Middle vertex (%f, %f, %f, %f)\n",verts[idx].x, verts[idx].y, verts[idx].z, verts[idx].w);
	//}
	if(FIRST_THREAD())	{
		printf("First vertex (%f, %f, %f, %f)\n",verts[idx].x, verts[idx].y, verts[idx].z, verts[idx].w);
		printf("First  transformed vertex (%f, %f, %f)\n",tempPos.x, tempPos.y, tempPos.z);
		printf("First  projected vertex (%f, %f, %f)\n",projTemp.x, projTemp.y, projTemp.z);
		printf("Vertex in frustum : ");
		printf(vertexInFrustum(tempPos) ? "true\n" : "false\n");
		//printDeviceMatrix(kinectProjectionMatrix);
	}
	float3 p = make_float3(tempPos);
	float3 pn = make_float3(normals[idx]);
	float3 rayStart = p - (d_hashtableParams.truncation * pn);
	float3 rayEnd = p + (d_hashtableParams.truncation * pn);

	//Now find their voxel blocks
	int3 startBlock = world2Block(rayStart);
	int3 endBlock = world2Block(rayEnd);
	float3 rayDir = normalize(rayEnd - rayStart);

	int3 step = make_int3(signbit(rayDir.x),signbit(rayDir.y),signbit(rayDir.z));	//block stepping size
	float3 next_boundary = (rayStart + make_float3(step));

	//calculate distance to next barrier
	float3 tMax = ((next_boundary - rayStart)) / rayDir;
	float3 tDelta = (voxSize / rayDir);
	tDelta.x *= step.x;
	tDelta.y *= step.y;
	tDelta.z *= step.z;

	//convert to voxel-blocks
	int3 idStart = world2Block(rayStart);
	int3 idEnd = world2Block(rayEnd);
	int3 temp = idStart;

	if (rayDir.x == 0.0f) { tMax.x = INF; tDelta.x = INF; }
	if (next_boundary.x - rayStart.x == 0.0f) { tMax.x = INF; tDelta.x = INF; }

	if (rayDir.y == 0.0f) { tMax.y = INF; tDelta.y = INF; }
	if (next_boundary.y - rayStart.y == 0.0f) { tMax.y = INF; tDelta.y = INF; }

	if (rayDir.z == 0.0f) { tMax.z = INF; tDelta.z = INF; }
	if (next_boundary.z - rayStart.z == 0.0f) { tMax.z = INF; tDelta.z = INF; }


	if(FIRST_THREAD())	{
		printf("Starting insertion now\n");
	}
	//first insert idStart block into the hashtable
	if(blockInFrustum(temp))	{
		//TODO : ease this later
		if(FIRST_THREAD())	{
			printf("Inserting block (%d, %d, %d)\n",temp.x, temp.y, temp.z);
			//printf("Block in frustum : ");
			//printf(blockInFrustum(temp) ? "true\n" : "false\n");
		}
		insertVoxelEntry(temp);
	}

	while((temp.x != idEnd.x) && (temp.y != idEnd.y) && (temp.z != idEnd.z))	{

		if(tMax.x < tMax.y && tMax.x < tMax.z)	{
			temp.x += step.x;
			//if(temp.x == idEnd.x) break;
			tMax.x += tDelta.x;
		}
		else if(tMax.z < tMax.y)	{
			temp.z += step.z;
			//if(temp.z == idEnd.z) break;
			tMax.z += tDelta.z;
		}
		else{
			temp.y += step.y;
			//if(temp.y == idEnd.y) break;
			tMax.y += tDelta.y;
		}

		//check if block is in view, then insert into table
		if(blockInFrustum(temp))	{
			if(FIRST_THREAD())	{
				printf("Inserting block (%d, %d, %d)\n",temp.x, temp.y, temp.z);
				//printf("Block in frustum : ");
				//printf(blockInFrustum(temp) ? "true\n" : "false\n");
			}
			bool status = insertVoxelEntry(temp);
			if(status && FIRST_THREAD())	{
				printf("Succsful insertion\n");
			}
		}
		//cout<<"\nVisited "<<glm::to_string(temp);
		//iter++;
	}
	//By now all necessary blocks should have been allocated
}

//! Allocate all hash blocks which are corresponding to depth map entries
extern "C" void allocBlocks(const float4* verts, const float4* normals)	{

	//const dim3 blocks(640/8, 480/8, 1);
	const dim3 blocks(1, 1, 1);
	const dim3 threads(8, 8, 1);
	std::cout<<"Running AllocBlocksKernel\n";
	allocBlocksKernel <<<blocks, threads>>>(verts, normals);
	checkCudaErrors(cudaDeviceSynchronize());
}

//! Generate a linear hash-array with only occupied entries
__global__
void flattenKernel()	{
	const int idx = (blockDim.x * blockIdx.x) + threadIdx.x;

	if(idx < (d_hashtableParams.bucketSize * d_hashtableParams.numBuckets)) return;

	__shared__ int localCounter;
	if(threadIdx.x == 0) localCounter = 0;
	__syncthreads();

	//local address within block
	int localAddr = -1;
	VoxelEntry& entry = ptrHldr.d_hashTable[idx];
	if(entry.ptr != FREE_BLOCK && blockInFrustum(entry.pos))	{
		localAddr = atomicAdd(&localCounter, 1);
	}
	__syncthreads();

	//update global count of occupied blocks
	__shared__ int globalAddr;
	if(threadIdx.x==0 && localCounter > 0)	{
		globalAddr = atomicAdd(ptrHldr.d_compactifiedHashCounter, localCounter);
	}
	__syncthreads();

	//assign local address and copy
	if(localAddr != -1)	{
		const unsigned int addr = globalAddr + localAddr;
		ptrHldr.d_compactifiedHashTable[addr] = entry;
	}
}

extern "C" int flattenIntoBuffer()	{
	//first set numOccupiedBlocks = 0
	//first clear previously flattened hashtable buffer
	checkCudaErrors(cudaMemset(ptrHldr.d_compactifiedHashTable, 0, sizeof(VoxelEntry) * d_hashtableParams.numOccupiedBlocks));
	checkCudaErrors(cudaMemset(ptrHldr.d_compactifiedHashCounter, 0, sizeof(int)));

	dim3 blocks = dim3((d_hashtableParams.numBuckets * d_hashtableParams.bucketSize + 256 -1)/256, 1, 1);
	dim3 threads = dim3(256, 1, 1);
	flattenKernel<<<blocks, threads>>>();
	checkCudaErrors(cudaDeviceSynchronize());
	int occupiedBlocks = 0;
	checkCudaErrors(cudaMemcpy(&occupiedBlocks, ptrHldr.d_compactifiedHashCounter, sizeof(int), cudaMemcpyDeviceToHost));

	return occupiedBlocks;
}

