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

}

SDF_Hashtable::SDF_Hashtable()	{
	HashTableParams h_tempParams;
	allocate(h_tempParams);
}

