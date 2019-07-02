#ifdef _WIN32
#include <windows.h>
#endif

#include "prereq.h"
#include "SDF_Hashtable.h"
#include "SDFRenderer.h"
#include "VoxelUtils.h"

void SDF_Hashtable::integrate(const float4x4& viewMat, const float4* verts, const float4* normals)	{

	mapGLobjectsToCUDApointers(numVisibleBlocks_res, compactHashtable_res, sdfBlocks_res);
	//first update the device HashParams variable
	float4x4 inv_global_transform = viewMat.getInverse();

	h_hashtableParams.global_transform = viewMat;
	h_hashtableParams.inv_global_transform = inv_global_transform;

	//upload to device
	updateConstantHashTableParams(h_hashtableParams);

	//TODO : reset hash-table mutexes
	resetHashTableMutexes(h_hashtableParams);

	//launch kernel to insert voxelentries into hashtable
	allocBlocks(verts, normals);

	//consolidate visible entries into a flat buffer
	int occupiedBlockCount = flattenIntoBuffer(h_hashtableParams);
	std::cout<<"occupiedBlockCount : "<<occupiedBlockCount<<"\n";
	h_hashtableParams.numOccupiedBlocks = occupiedBlockCount;
	updateConstantHashTableParams(h_hashtableParams);

	//integrate vertices into SDF volume
	integrateDepthMap(h_hashtableParams, verts);
	std::cout << "depth map integrated into volume! \n";

	unmapCUDApointers();
}

void SDF_Hashtable::registerGLtoCUDA(SDFRenderer& renderer) {
	//rendererRef = &renderer;
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&numVisibleBlocks_res, renderer.numOccupiedBlocks_handle, cudaGraphicsRegisterFlagsNone));
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&compactHashtable_res, renderer.compactHashTable_handle, cudaGraphicsRegisterFlagsNone));
	checkCudaErrors(cudaGraphicsGLRegisterBuffer(&sdfBlocks_res, renderer.SDF_VolumeBuffer_handle, cudaGraphicsRegisterFlagsNone));
	//int numVisibleBlocks_handle = renderer.numOccupiedBlocks_handle;
	//mapGLobjectsToCUDApointers(*rendererRef);
	std::cout << "GL resources registered to CUDA hashtable\n";
}

void SDF_Hashtable::unmapCUDApointers()
{
	checkCudaErrors(cudaGraphicsUnmapResources(1, &numVisibleBlocks_res, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &compactHashtable_res, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &sdfBlocks_res, 0));
	std::cout << "CUDA device pointers unmapped!\n";
}

SDF_Hashtable::SDF_Hashtable()	{
	h_hashtableParams.numBuckets = 500000;
	h_hashtableParams.bucketSize = 5;	//10
	h_hashtableParams.attachedLinkedListSize = 4;	//7
	h_hashtableParams.numVoxelBlocks = 100000;
	h_hashtableParams.voxelBlockSize = 8;
	h_hashtableParams.voxelSize = 0.05f;
	h_hashtableParams.numOccupiedBlocks = 0;
	h_hashtableParams.maxIntegrationDistance = 4.0f;
	h_hashtableParams.truncScale = 0.01f;
	h_hashtableParams.truncation = 0.02f;
	h_hashtableParams.integrationWeightSample = 10;
	h_hashtableParams.integrationWeightMax = 255;

	updateConstantHashTableParams(h_hashtableParams);
	deviceAllocate(h_hashtableParams);
	std::cout<<"GPU memory allocated\n";

	calculateKinectProjectionMatrix();
	std::cout<<"Kinect projection matrix calculated\n";
}

SDF_Hashtable::~SDF_Hashtable()	{
	deviceFree();
	std::cout<<"GPU memory freed\n";
}

