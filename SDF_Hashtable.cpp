#include "SDF_Hashtable.h"
#include "VoxelUtils.h"

void SDF_Hashtable::integrate(const float4x4& viewMat, const float4* verts, const float4* normals)	{
	//first update the device HashParams variable
	HashTableParams h_hashtableParams;
	float4x4 inv_global_transform = viewMat.getInverse();

	h_hashtableParams.global_transform = viewMat;
	h_hashtableParams.inv_global_transform = inv_global_transform;

	//upload to device
	updateConstantHashTableParams(h_hashtableParams);

	//launch kernel to insert voxelentries into hashtable
	allocBlocks(verts, normals);

	//consolidate visible entries into a flat buffer
	int occupiedBlockCount = flattenIntoBuffer();
	std::cout<<"occupiedBlockCount : "<<occupiedBlockCount<<"\n";
	h_hashtableParams.numOccupiedBlocks = occupiedBlockCount;
	updateConstantHashTableParams(h_hashtableParams);

}

SDF_Hashtable::SDF_Hashtable()	{
	HashTableParams h_tempParams;
	h_tempParams.numBuckets = 500000;
	h_tempParams.bucketSize = 10;
	h_tempParams.attachedLinkedListSize = 7;
	h_tempParams.numVoxelBlocks = 100000;
	h_tempParams.voxelBlockSize = 8;
	h_tempParams.voxelSize = 0.05f;
	h_tempParams.numOccupiedBlocks = 0;
	h_tempParams.maxIntegrationDistance = 4.0f;
	h_tempParams.truncScale = 0.01f;
	h_tempParams.truncation = 0.02f;
	h_tempParams.integrationWeightSample = 10;
	h_tempParams.integrationWeightMax = 255;

	deviceAllocate(h_tempParams);
	std::cout<<"GPU memory allocated\n";
}

SDF_Hashtable::~SDF_Hashtable()	{
	deviceFree();
	std::cout<<"GPU memory freed\n";
}

