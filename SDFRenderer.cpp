#include "common.h"
#include "SDFRenderer.h"
#include "VoxelDataStructures.h"
#include <vector>

SDFRenderer::SDFRenderer() {

	raycast_shader = std::unique_ptr<ShaderProgram>(new ShaderProgram());
	raycast_shader->initFromFiles("shaders/drawBox.vert", "shaders/drawBox.geom", "shaders/drawBox.frag");
	raycast_shader->addAttribute("voxentry");
	raycast_shader->addUniform("viewMat");
	raycast_shader->addUniform("projMat");

	//init GL resources 
	glGenVertexArrays(1, &SDF_VAO);
	glBindVertexArray(SDF_VAO);

	glGenBuffers(1, &numOccupiedBlocks_handle);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, numOccupiedBlocks_handle);
	glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(unsigned int), &numOccupiedBlocks, GL_STATIC_COPY);

	glGenBuffers(1, &compactHashTable_handle);
	glBindBuffer(GL_ARRAY_BUFFER, compactHashTable_handle);
	glBufferData(GL_ARRAY_BUFFER, sizeof(VoxelEntry) * numBuckets * bucketSize , nullptr, GL_DYNAMIC_DRAW);
	glEnableVertexAttribArray(raycast_shader->attribute("voxentry"));
	glVertexAttribPointer(raycast_shader->attribute("voxentry"), 3, GL_INT, GL_FALSE, sizeof(VoxelEntry), 0);

	glGenBuffers(1, &SDF_VolumeBuffer_handle);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, SDF_VolumeBuffer_handle);
	glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(Voxel) * voxelBlockSize * voxelBlockSize * voxelBlockSize * numVoxelBlocks, nullptr, GL_STATIC_COPY);

	//unbind
	glBindVertexArray(0);
	glBindBuffer(GL_SHADER_STORAGE_BLOCK, 0);

	//register with SDF_Hashtable class
}

//just to check if memory is mapped correctly. TODO : remove later
void SDFRenderer::printSDFdata() {
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, numOccupiedBlocks_handle);
	glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(unsigned int), &numOccupiedBlocks);
	std::vector<VoxelEntry> vox_entries;
	//now copy back VoxelEntries
	vox_entries.resize(numOccupiedBlocks);
	glBindBuffer(GL_ARRAY_BUFFER, compactHashTable_handle);
	glGetBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(VoxelEntry) * numOccupiedBlocks, vox_entries.data());
	
	std::vector<Voxel> sdf_chunks;
	sdf_chunks.resize(numOccupiedBlocks * 512);
	glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(Voxel) * 512 * numOccupiedBlocks, sdf_chunks.data());
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	std::cout << "numOccupiedBlocks from GL :" << numOccupiedBlocks << "\n";
	for (auto &i : vox_entries) {
		std::cout << "pos : (" << i.pos.x << ", " << i.pos.y << ", " << i.pos.z << ") ptr = " << i.ptr << " offset = " << i.offset << "\n";
	}
	std::cout <<"\n\n\n SDFs \n";
	for(int i = 0; i < numOccupiedBlocks; ++i) {
		std::cout << i << ") : ";
		for (int j = 0; j < 512; ++j) {
			std::cout << sdf_chunks[i * 512 + j].sdf << "\t";
		}
		std::cout << "\n\n\n";
	}

}

void SDFRenderer::render(const glm::mat4& viewMat) {
	glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
	glBindVertexArray(SDF_VAO);
	raycast_shader->use();
	glUniformMatrix4fv(raycast_shader->uniform("viewMat"), 1, false, glm::value_ptr(viewMat));
	glUniformMatrix4fv(raycast_shader->uniform("projMat"), 1, false, glm::value_ptr(projMat));
	glDrawArrays(GL_POINTS, 0, numOccupiedBlocks);
	glBindVertexArray(0);

	//raycast_shader->uniform("viewMat")

}

SDFRenderer::~SDFRenderer() {

}