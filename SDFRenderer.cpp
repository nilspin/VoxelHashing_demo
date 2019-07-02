#include "common.h"
#include "SDFRenderer.h"
#include "VoxelDataStructures.h"

SDFRenderer::SDFRenderer() {
	//init GL resources 
	glGenVertexArrays(1, &SDF_VAO);
	glBindVertexArray(SDF_VAO);

	glGenBuffers(1, &numOccupiedBlocks_handle);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, numOccupiedBlocks_handle);
	glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(unsigned int), &numOccupiedBlocks, GL_STATIC_COPY);

	glGenBuffers(1, &compactHashTable_handle);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, compactHashTable_handle);
	glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(VoxelEntry) * numBuckets * bucketSize , nullptr, GL_STATIC_COPY);

	glGenBuffers(1, &SDF_VolumeBuffer_handle);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, SDF_VolumeBuffer_handle);
	glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(Voxel) * voxelBlockSize * voxelBlockSize * voxelBlockSize * numVoxelBlocks, nullptr, GL_STATIC_COPY);

	//unbind
	glBindVertexArray(0);
	glBindBuffer(GL_SHADER_STORAGE_BLOCK, 0);

	//register with SDF_Hashtable class
}

void SDFRenderer::render(const glm::mat4& viewMat) {

}

SDFRenderer::~SDFRenderer() {

}