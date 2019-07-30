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
__constant__ PtrContainer d_ptrHldr;
PtrContainer h_ptrHldr;


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
//Voxel *d_SDFBlocks;
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

void updateDevicePointers() {
	size_t size;
	checkCudaErrors(cudaGetSymbolSize(&size, d_ptrHldr));
	checkCudaErrors(cudaMemcpyToSymbol(d_ptrHldr, &h_ptrHldr, size, 0, cudaMemcpyHostToDevice));
	std::cout << "h_ptrHldr supposedly copied to device\n";
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

//TODO : Confusion here. FInish this later
//no GL functions should be called after pointers are mapped to cuda
//__host__
extern "C" void mapGLobjectsToCUDApointers(cudaGraphicsResource* numBlocks_res, cudaGraphicsResource* compactHashtable_res,
	cudaGraphicsResource* sdfVolume_res) {

	size_t returnedBufferSize;
	checkCudaErrors(cudaGraphicsMapResources(1, &numBlocks_res, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&h_ptrHldr.d_compactifiedHashCounter, &returnedBufferSize, numBlocks_res));
	int occupiedBlocks = -1;
	//TODO : following line is just a check. remove it later
	checkCudaErrors(cudaMemcpy(&occupiedBlocks, &h_ptrHldr.d_compactifiedHashCounter[0], sizeof(int), cudaMemcpyDeviceToHost));
	std::cout << "(after GL mapping)numVisibleBlocks : " << occupiedBlocks << "\n";

	checkCudaErrors(cudaGraphicsMapResources(1, &compactHashtable_res, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&h_ptrHldr.d_compactifiedHashTable, &returnedBufferSize, compactHashtable_res));
	std::cout << "(after GL mapping) size of compactifiedHashtable(in bytes) : " << returnedBufferSize<< "\n";

	checkCudaErrors(cudaGraphicsMapResources(1, &sdfVolume_res, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&h_ptrHldr.d_SDFBlocks, &returnedBufferSize, sdfVolume_res));
	std::cout << "(after GL mapping) size of SDFVoxelBlocks(in bytes) : " << returnedBufferSize << "\n";
	updateDevicePointers();

}

//__host__
extern "C" void resetHashTableMutexes(const HashTableParams& params) {
	checkCudaErrors(cudaMemset(h_ptrHldr.d_hashTableBucketMutex, 0, sizeof(int)*params.numBuckets));
	updateDevicePointers();
}

__global__
void resetHashTableKernel(VoxelEntry* table) {
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= d_hashtableParams.bucketSize*d_hashtableParams.numBuckets) return;
	table[idx].offset = NO_OFFSET;
	table[idx].ptr = FREE_BLOCK;
	table[idx].pos = make_int3(INF, INF, INF);
}

//Call this only once!!!
__global__
void resetHeapKernel(unsigned int* d_heap) {
	const int idx = blockIdx.x*blockDim.x + threadIdx.x;
	if (idx >= d_hashtableParams.numVoxelBlocks) return;
	d_heap[idx] = idx;
}

//TODO do this using thrust instead
__host__
void deviceAllocate(const HashTableParams &params)	{
	//PtrContainer h_ptrHldr;
	checkCudaErrors(cudaMalloc((void**)&h_ptrHldr.d_heap, sizeof(unsigned int) * params.numVoxelBlocks));
	checkCudaErrors(cudaMalloc((void**)&h_ptrHldr.d_hashTable, sizeof(VoxelEntry) * params.numBuckets * params.bucketSize));
	//checkCudaErrors(cudaMalloc((void**)&h_ptrHldr.d_compactifiedHashTable, sizeof(VoxelEntry) * params.numBuckets * params.bucketSize));
	checkCudaErrors(cudaMalloc((void**)&h_ptrHldr.d_hashTableBucketMutex, sizeof(int) * params.numBuckets));
	//checkCudaErrors(cudaMalloc((void**)&h_ptrHldr.d_SDFBlocks, sizeof(Voxel) * params.numVoxelBlocks * params.voxelBlockSize * params.voxelBlockSize * params.voxelBlockSize));
	checkCudaErrors(cudaMalloc((void**)&h_ptrHldr.d_heapCounter, sizeof(int)));
	//checkCudaErrors(cudaMalloc((void**)&h_ptrHldr.d_compactifiedHashCounter, sizeof(int)));	//TODO : remove this

	updateDevicePointers();

	//init with correct values
	const int totalThreads = params.numBuckets*params.bucketSize;
	int blocks = (totalThreads / 1024) + 1;
	int threads = 1024;
	//init buffers with default values. TODO: Launch following two kernels asynchronously
	resetHashTableKernel<<<blocks, threads >>> (h_ptrHldr.d_hashTable);
	checkCudaErrors(cudaDeviceSynchronize());
	//TODO : reset compactifiedHashTable after registering GL buffers
	//resetHashTableKernel<<<blocks, threads >>> (h_ptrHldr.d_compactifiedHashTable);
	//checkCudaErrors(cudaDeviceSynchronize());

	int heapBlocks = (params.numVoxelBlocks / threads) + 1;
	resetHeapKernel<<<heapBlocks, threads>>>(h_ptrHldr.d_heap);
	checkCudaErrors(cudaDeviceSynchronize());

	//set rest of data 0
	//checkCudaErrors(cudaMemset(h_ptrHldr.d_heap, 0, sizeof(int)*params.numVoxelBlocks));	//don't need this anymore
	checkCudaErrors(cudaMemset(h_ptrHldr.d_hashTableBucketMutex, 0, sizeof(int)*params.numBuckets));
	//TODO : reset SDFBlocks after registering GL buffers
	//checkCudaErrors(cudaMemset(h_ptrHldr.d_SDFBlocks, 0, sizeof(Voxel) * params.numVoxelBlocks *
		//params.voxelBlockSize * params.voxelBlockSize * params.voxelBlockSize));
	checkCudaErrors(cudaMemset(h_ptrHldr.d_heapCounter, 0, sizeof(int)));
	//checkCudaErrors(cudaMemset(h_ptrHldr.d_compactifiedHashCounter, 0, sizeof(int)));	//TODO : remove this

	//set d_heapCounter = numVoxelBlocks -1;
	int heapCounterInitVal = params.numVoxelBlocks - 1;
	checkCudaErrors(cudaMemcpy(&h_ptrHldr.d_heapCounter[0], &heapCounterInitVal, sizeof(int), cudaMemcpyHostToDevice));
	//now copy this struct back to device
	updateDevicePointers();
}

__host__
void deviceFree()	{
	checkCudaErrors(cudaFree(d_ptrHldr.d_hashTable));
	checkCudaErrors(cudaFree(d_ptrHldr.d_heap));
	checkCudaErrors(cudaFree(d_ptrHldr.d_heapCounter));
	//checkCudaErrors(cudaFree(d_ptrHldr.d_compactifiedHashTable));
	//checkCudaErrors(cudaFree(d_ptrHldr.d_compactifiedHashCounter));	//TODO : remove this
	//checkCudaErrors(cudaFree(d_ptrHldr.d_SDFBlocks));
	checkCudaErrors(cudaFree(d_ptrHldr.d_hashTableBucketMutex));
}

__host__
void calculateKinectProjectionMatrix()	{

	float3x3 m(intrinsicsTranspose);
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
	//float3 vx = make_float3(voxel);
	//int x = __float2int_rz(vx.x / size);
	//int y = __float2int_rz(vx.y / size);
	//int z = __float2int_rz(vx.z / size);
	//return make_int3(x, y, z);
	if(voxel.x < 0) voxel.x -= size-1;	//i.e voxelBlockSize -1
	if(voxel.y < 0) voxel.y -= size-1;
	if(voxel.z < 0) voxel.z -= size-1;
	return make_int3(voxel.x/size, voxel.y/size, voxel.z/size);
}

__device__
int3 world2Voxel(const float3& point)	{
	const float size = d_hashtableParams.voxelSize;
	float3 p = point/size;
	int3 centerOffset = make_int3(copysignf(1, p.x), copysignf(1, p.y), copysignf(1, p.z));
	int3 voxelPos =  make_int3(p + make_float3(centerOffset.x*0.5, centerOffset.y*0.5, centerOffset.z*0.5));//return center
	return voxelPos;
}

__device__
int3 block2Voxel(const int3& block)	{
	int3 voxelPos = make_int3(block.x, block.y, block.z) * d_hashtableParams.voxelBlockSize;
	return voxelPos;
}

__device__
float3 voxel2World(const int3& voxel)	{
	float3 worldPos = make_float3(voxel) * d_hashtableParams.voxelSize;
	return worldPos;
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
int3 delinearizeVoxelPos(const unsigned int index)	{
	const int size = d_hashtableParams.voxelBlockSize;
	unsigned int x = index % size;
	unsigned int y = (index % (size * size)) / size;
	unsigned int z = index / (size * size);
	return make_int3(x,y,z);
}

__inline__ __device__
int allocSingleBlockInHeap()	{	//int ptr
	//decrement total available blocks by 1
	int addr = atomicSub(&d_ptrHldr.d_heapCounter[0], 1);	//TODO: make this uint
	//if (addr < 0) return -1;	//negative index shouldn't, but still happens :(
	return d_ptrHldr.d_heap[addr];
}

__device__
void removeSingleBlockInHeap(int ptr)	{
	//int delIdx = ptr / 512;
	int addr = atomicAdd(&d_ptrHldr.d_heapCounter[0], 1);	//TODO: make this uint
	d_ptrHldr.d_heap[addr + 1] = ptr;
}

//Frustum culling
__inline__ __device__
bool blockInFrustum(int3 blockId) {
	float3 worldPos = block2World(blockId);
	float4 pos = make_float4(worldPos.x, worldPos.y, worldPos.z, 1);
	pos = d_hashtableParams.global_transform * pos;	//TODO : shouldn't this be inv_global_transform?
	float3 projected = make_float3(pos.x, pos.y, pos.z);
	projected = kinectProjectionMatrix * projected;
	projected = projected / projected.z;

	int x = __float2int_rz(projected.x);
	int y = __float2int_rz(projected.y);
	if (x < 640 && x >= 0 && y < 480 && y >= 0) {
		return true;
	}
	return false;
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
		VoxelEntry& curr = d_ptrHldr.d_hashTable[startIndex + i];
		if((curr.pos.x == pos.x) && (curr.pos.y == pos.y) &&(curr.pos.z == pos.z)
				&& (curr.ptr != FREE_BLOCK)) {
			return curr;
		}
	}

#ifdef LINKED_LIST_ENABLED

	//[2] block not found. handle collisions by traversing tail linked list
	const int lastEntryInBucket = (hash+1)*bucketSize -1;
	i = lastEntryInBucket;
	//memorize idx at list end and memorize offset from last
	//element of bucket to list end
	int iter = 0;
	const int maxIter = d_hashtableParams.attachedLinkedListSize;
	while(iter < maxIter)	{

		VoxelEntry curr = d_ptrHldr.d_hashTable[i];
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

#endif // LINKED_LIST_ENABLED

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
		int idx = startIndex+i;
		idx = idx % (numBuckets * bucketSize);
		VoxelEntry &curr = d_ptrHldr.d_hashTable[idx];
		if(curr.pos.x == data.x && curr.pos.y == data.y && curr.pos.z == data.z
				&& curr.ptr != FREE_BLOCK)	return false;
		if(curr.ptr == FREE_BLOCK)	{
			//TODO shouldn't the following be [hash] instead of [idx] ?
			int prevVal = atomicExch(&d_ptrHldr.d_hashTableBucketMutex[hash], LOCKED_BLOCK);
			if(prevVal != LOCKED_BLOCK)	{	//means we can lock current bucket
			//{
				curr.pos = data;
				curr.offset = NO_OFFSET;
				int ptrIdx = allocSingleBlockInHeap() * 512;
				if (ptrIdx < 0)	return false;	//all VoxelBlocks occupied
				curr.ptr = ptrIdx;
				printf("Inserted block : (%d, %d, %d) at idx %d\n", data.x, data.y, data.z, ptrIdx/512);
				return true;
			}
		}
	}

#ifdef LINKED_LIST_ENABLED

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
		VoxelEntry& curr = d_ptrHldr.d_hashTable[i];
		if(curr.ptr != FREE_BLOCK)	{
			if(curr.pos.x == data.x && curr.pos.y == data.y &&
					curr.pos.z == data.z && curr.ptr != FREE_BLOCK)	{
				return false;	//alloc unsuccessful because block already there
			}
			if(curr.offset == 0)	{//end of list, lookahead till we find empty slot
				int j=1;
				//[1] lock the parent block
				int prevVal = atomicExch(&d_ptrHldr.d_hashTableBucketMutex[hash],
						LOCKED_BLOCK);
				if(prevVal != LOCKED_BLOCK)	{//if we got the lock
					//[2] then lookahead for empty block in new bucket
					while(j<10)	{
						if(d_ptrHldr.d_hashTable[i+j].ptr == FREE_BLOCK)	break;
						j++;
					}
					if(j==10)	{
						//we couldn't find empty space despite looking ahead 10 spaces
						return false;
					}
					//[3] now lock this new bucket and insert the block
					prevVal = atomicExch(&d_ptrHldr.d_hashTableBucketMutex[(i+j)/numBuckets],
							LOCKED_BLOCK);
					if(prevVal != LOCKED_BLOCK)	{
						VoxelEntry& next = d_ptrHldr.d_hashTable[i+j];
						//TODO maybe we can do away with this check
						if(next.ptr == FREE_BLOCK)	{
							int ptrIdx = allocSingleBlockInHeap() * 512;
							if (ptrIdx < 0)	return false;
							next.ptr = ptrIdx;
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
					if(d_ptrHldr.d_hashTable[j].ptr == FREE_BLOCK)	{
						//[a] free space found. first lock bucket with curr
						int prevVal = atomicExch(&d_ptrHldr.d_hashTableBucketMutex[hash/numBuckets], LOCKED_BLOCK);
						if(prevVal != LOCKED_BLOCK)	{
							//[b] then lock bucket with new space
							prevVal = atomicExch(&d_ptrHldr.d_hashTableBucketMutex[j/numBuckets], LOCKED_BLOCK);
							if(prevVal != LOCKED_BLOCK)	{
								VoxelEntry& ins = d_ptrHldr.d_hashTable[j];
								ins.offset = i + curr.offset - j;
								int ptrIdx = allocSingleBlockInHeap() * 512;
								if (ptrIdx < 0)	return false;
								ins.ptr = ptrIdx;
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
#endif // LINKED_LIST_ENABLED

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
		VoxelEntry &curr = d_ptrHldr.d_hashTable[idx];
		if(curr.pos.x == data.x && curr.pos.y == data.y && curr.pos.z == data.z
				&& curr.ptr != FREE_BLOCK)	{return false;}
		if(curr.ptr == FREE_BLOCK)	{
			//TODO shouldn't the following be [hash] instead of [idx] ?
			//try locking current bucket
			int prevVal = atomicExch(&d_ptrHldr.d_hashTableBucketMutex[hash], LOCKED_BLOCK);
			if(prevVal != LOCKED_BLOCK)	{	//means we can lock current bucket
				curr.pos = make_int3(0);
				curr.offset = NO_OFFSET;
				removeSingleBlockInHeap(curr.ptr/512);
				curr.ptr = FREE_BLOCK;
				return true;
			}
		}
	}

#ifdef LINKED_LIST_ENABLED

	//deletion in linked list
	int lastEntry = beforeThis(data);
	if(lastEntry == -1)	{return false;}	//error
	VoxelEntry& prev = d_ptrHldr.d_hashTable[lastEntry];
	VoxelEntry& curr = d_ptrHldr.d_hashTable[lastEntry + prev.offset];
	//lock the bucket with curr
	int prevVal = atomicExch(&d_ptrHldr.d_hashTableBucketMutex[hash], LOCKED_BLOCK);
	if(prevVal!=LOCKED_BLOCK)	{	//lock acquired
		prevVal = atomicExch(&d_ptrHldr.d_hashTableBucketMutex[lastEntry / numBuckets],
				LOCKED_BLOCK);
		if(prevVal != LOCKED_BLOCK)	{
			//TODO FINISH THIS!!!
			prev.offset += curr.offset;
			curr.pos = make_int3(0);
			curr.offset = NO_OFFSET;
			removeSingleBlockInHeap(curr.ptr/512);
			curr.ptr = FREE_BLOCK;
		}
	}

	return false;//delete didn't happen :(

#endif // LINKED_LIST_ENABLED

}

__global__
void allocBlocksKernel(const float4* verts, const float4* normals)	{	//Do we need normal data here?

	const float voxelSize = d_hashtableParams.voxelSize;
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
	//float3 projTemp = make_float3(tempPos.x, tempPos.y, tempPos.z);
	//projTemp = kinectProjectionMatrix * projTemp;
	//projTemp = projTemp/projTemp.z;
	//TODO : Erase this later
	//if(idx==153600)	{
	//	printf("Middle vertex (%f, %f, %f, %f)\n",verts[idx].x, verts[idx].y, verts[idx].z, verts[idx].w);
	//}
	float3 p = make_float3(tempPos);
	float3 pn = make_float3(normals[idx]);
	float3 rayStart = p;// -(d_hashtableParams.truncation * pn);
	//float3 rayEnd = p + (d_hashtableParams.truncation * pn);

	//Now find their voxel blocks
	int3 startBlock = world2Block(rayStart);
	/*TODO : Maybe we don't need this???
	int3 endBlock = world2Block(rayEnd);
	float3 rayDir = normalize(rayEnd - rayStart);

	int3 step = make_int3(signbit(rayDir.x),signbit(rayDir.y),signbit(rayDir.z));	//block stepping size
	float3 next_boundary = (rayStart + make_float3(step));

	//calculate distance to next barrier
	float3 tMax = ((next_boundary - rayStart)) / rayDir;
	float3 tDelta = (voxelSize / rayDir);
	tDelta.x *= step.x;
	tDelta.y *= step.y;
	tDelta.z *= step.z;

	//convert to voxel-blocks
	int3 idStart = world2Block(rayStart);
	float3 wrldPos = block2World(idStart);
	//printf("Vertex = (%f, %f, %f) idStart : (%d, %d, %d) and back (%f, %f, %f)\n",verts[idx].x, verts[idx].y, verts[idx].z, idStart.x, idStart.y, idStart.z, wrldPos.x, wrldPos.y, wrldPos.z);

	int3 idEnd = world2Block(rayEnd);
	int3 temp = idStart;

	if (rayDir.x == 0.0f) { tMax.x = INF; tDelta.x = INF; }
	if (next_boundary.x - rayStart.x == 0.0f) { tMax.x = INF; tDelta.x = INF; }

	if (rayDir.y == 0.0f) { tMax.y = INF; tDelta.y = INF; }
	if (next_boundary.y - rayStart.y == 0.0f) { tMax.y = INF; tDelta.y = INF; }

	if (rayDir.z == 0.0f) { tMax.z = INF; tDelta.z = INF; }
	if (next_boundary.z - rayStart.z == 0.0f) { tMax.z = INF; tDelta.z = INF; }


	//first insert idStart block into the hashtable
	//bool status = vertexInFrustum(tempPos);
	*/
	//TODO : NOTE these blockInFrustum checks are useless since having depth data means vertex is visible
	if(blockInFrustum(startBlock))	{	//blockInFrustum(temp)
		insertVoxelEntry(startBlock);
	}	//, instead simply
	//insertVoxelEntry(temp);

	/*	TODO: Maybe we don't need this???
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
			bool status = insertVoxelEntry(temp);
		}
		//iter++;
	}
	*/
	//By now all necessary blocks should have been allocated
}

//! Allocate all hash blocks which are corresponding to depth map entries
extern "C" void allocBlocks(const float4* verts, const float4* normals)	{

	const dim3 blocks(640/16, 480/16, 1);
	//const dim3 blocks(1, 1, 1);
	const dim3 threads(16, 16, 1);
	std::cout<<"Running AllocBlocksKernel\n";
	allocBlocksKernel <<<blocks, threads>>>(verts, normals);
	checkCudaErrors(cudaDeviceSynchronize());
}

//! Generate a linear hash-array with only occupied entries
__global__
void flattenKernel()	{
	const int idx = (blockDim.x * blockIdx.x) + threadIdx.x;

	if(idx >= (d_hashtableParams.bucketSize * d_hashtableParams.numBuckets)) return;

	__shared__ int localCounter;
	if(threadIdx.x == 0) localCounter = 0;
	__syncthreads();

	//local address within block
	int localAddr = -1;
	const VoxelEntry& entry = d_ptrHldr.d_hashTable[idx];
	if(entry.ptr != FREE_BLOCK && blockInFrustum(entry.pos))	{
		localAddr = atomicAdd(&localCounter, 1);
	}
	__syncthreads();

	//update global count of occupied blocks
	__shared__ int globalAddr;
	if(threadIdx.x==0 && localCounter > 0)	{
		globalAddr = atomicAdd(&d_ptrHldr.d_compactifiedHashCounter[0], localCounter);
	}
	__syncthreads();

	//assign local address and copy
	if(localAddr != -1)	{
		const unsigned int addr = globalAddr + localAddr;
		d_ptrHldr.d_compactifiedHashTable[addr] = entry;
	}
}

extern "C" int flattenIntoBuffer(const HashTableParams& params)	{
	//first set numOccupiedBlocks = 0
	//first clear previously flattened hashtable buffer
	const int totalThreads = params.numBuckets*params.bucketSize;
	int blocks = (totalThreads / 1024) + 1;
	int threads = 1024;
	//TODO : Do we really need to reset compactifiedHashTable? wouldn't it get overwritten by flattenKernel anyways?
	resetHashTableKernel <<<blocks, threads >>> (h_ptrHldr.d_compactifiedHashTable);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemset(h_ptrHldr.d_compactifiedHashCounter, 0, sizeof(int)));

	flattenKernel<<<blocks, threads>>>();
	checkCudaErrors(cudaDeviceSynchronize());
	int occupiedBlocks = 0;
	checkCudaErrors(cudaMemcpy(&occupiedBlocks, &h_ptrHldr.d_compactifiedHashCounter[0], sizeof(int), cudaMemcpyDeviceToHost));

	return occupiedBlocks;
}

__inline__ __device__
int2 project(float3 voxelWorldPos) {
 	//int3 pos = make_int3(point.x, point.y, point.z);
	//float3 worldPos = voxel2World(voxel);
	voxelWorldPos = kinectProjectionMatrix * voxelWorldPos;
	voxelWorldPos = voxelWorldPos / voxelWorldPos.z;
	return make_int2(voxelWorldPos.x, voxelWorldPos.y);
}

__inline __device__
Voxel combineVoxel(const Voxel& oldVox, const Voxel& currVox) {
	//TODO: add color later
	Voxel newVox;
	newVox.sdf = ((oldVox.sdf * (float)oldVox.weight) + (currVox.sdf * (float)currVox.weight)) / ((float)oldVox.weight + (float)currVox.weight);
	newVox.weight = min(d_hashtableParams.integrationWeightMax, (unsigned int)oldVox.weight + (unsigned int)currVox.weight);

	return newVox;
}

//Implementation of Curless & Levoy paper(1996)
__global__
void integrateDepthMapKernel(const float4* verts) {
	const VoxelEntry& entry = d_ptrHldr.d_compactifiedHashTable[blockIdx.x];
	int3 base_voxel = block2Voxel(entry.pos);

	uint i = threadIdx.x;
	int3 curr_voxel = base_voxel + delinearizeVoxelPos(i);
	float4 curr_voxel_float = make_float4(curr_voxel.x, curr_voxel.y, curr_voxel.z, 1.0);
	curr_voxel_float = d_hashtableParams.inv_global_transform * curr_voxel_float;
	curr_voxel = make_int3(curr_voxel_float.x, curr_voxel_float.y, curr_voxel_float.z);
	float3 voxel_worldPos = voxel2World(curr_voxel);
	int2 screenPos = project(voxel_worldPos);

	if ((screenPos.x < 0) || (screenPos.x >= 640) || (screenPos.y < 0) || (screenPos.y >= 480)) return;
	const int idx = (screenPos.y * 640) + screenPos.x;
	float depth = verts[idx].z;
	if (depth <= 0)	return;

	//TODO : define these explicitly, somewhere safe outside of here!
	float depthRangeMin = 0.5;	//metres
	float depthRangeMax = 5.0;
	float depthZeroOne = (depth - depthRangeMin) / (depthRangeMax - depthRangeMin);	//normalize current depth

	float sdf = depth - voxel_worldPos.z;
	float truncation = d_hashtableParams.truncation;	// +(d_hashtableParams.truncScale*depth);
	//i.e calculate truncation of the SDF for given depth value

	if (sdf > -truncation) {
		if (sdf >= 0) {
			sdf = fminf(truncation, sdf);
		}
		else {
			sdf = fmaxf(-truncation, sdf);
		}

		//Sets updation weight based on sensor noise. Farther depths have less weight. Copied from prof. Niessner's implementation
		//float weightUpdate = fmaxf(d_hashtableParams.integrationWeightSample * 1.5 * (1.0 - depthZeroOne), 1.0f);
		unsigned int weightUpdate = 10;	//let's keep this constant for now

		Voxel curr;
		curr.sdf = sdf;
		curr.weight = weightUpdate;
		//curr.color = make_uchar3(0, 255, 0);	//TODO : later

		const int oldVoxIdx = entry.ptr + i;

		Voxel fusedVoxel = combineVoxel(d_ptrHldr.d_SDFBlocks[oldVoxIdx], curr);
		d_ptrHldr.d_SDFBlocks[oldVoxIdx] = fusedVoxel;	//replace old voxel with new fused one
	}
}

extern "C" void integrateDepthMap(const HashTableParams& params, const float4* verts) {
	int threads = params.voxelBlockSize * params.voxelBlockSize * params.voxelBlockSize;
	int blocks = params.numOccupiedBlocks;

	if (params.numOccupiedBlocks > 0) {
		integrateDepthMapKernel<<<blocks, threads>>>(verts);
		checkCudaErrors(cudaDeviceSynchronize());
	}
}
