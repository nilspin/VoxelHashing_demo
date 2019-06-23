#ifdef _WIN32
#include <windows.h>
#endif

#include "SDF_Hashtable.h"
#include "VoxelUtils.h"

void SDF_Hashtable::integrate(const float4x4& viewMat, const float4* verts, const float4* normals)	{
	//first update the device HashParams variable
	float4x4 inv_global_transform = viewMat.getInverse();

	h_hashtableParams.global_transform = viewMat;
	h_hashtableParams.inv_global_transform = inv_global_transform;

	//upload to device
	updateConstantHashTableParams(h_hashtableParams);

	//launch kernel to insert voxelentries into hashtable
	allocBlocks(verts, normals);

	//consolidate visible entries into a flat buffer
	int occupiedBlockCount = flattenIntoBuffer(h_hashtableParams);
	std::cout<<"occupiedBlockCount : "<<occupiedBlockCount<<"\n";
	h_hashtableParams.numOccupiedBlocks = occupiedBlockCount;
	updateConstantHashTableParams(h_hashtableParams);

}

SDF_Hashtable::SDF_Hashtable()	{
	h_hashtableParams.numBuckets = 500000;
	h_hashtableParams.bucketSize = 10;
	h_hashtableParams.attachedLinkedListSize = 7;
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

