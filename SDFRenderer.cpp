#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image_write.h"
#include "common.h"
#include "SDFRenderer.h"
#include "SDFRendererUtils.h"
#include "VoxelDataStructures.h"
#include <vector>
#include <fstream>
#include <iomanip>
#include <iostream>

#define BUFFER_OFFSET(i) ((char*)NULL + (i))

SDFRenderer::SDFRenderer() {

	fbo_front = setupFBO_w_intTex();

	fbo_back = setupFBO_w_intTex();

	//raycast_shader = setupRaycastShader();

	depthWriteShader = setupDepthWriteShader();

	//---------------Debug---------------
	debug_ssbo = setup_Debug_SSBO();
	//-----------------------------------

	/*----------VAO-------------------*/
	//init GL resources
	glGenVertexArrays(1, &SDF_VAO);
	glBindVertexArray(SDF_VAO);

	glGenBuffers(1, &compactHashTable_handle);
	glBindBuffer(GL_ARRAY_BUFFER, compactHashTable_handle);
	glBufferData(GL_ARRAY_BUFFER, sizeof(VoxelEntry) * numBuckets * bucketSize, nullptr, GL_STATIC_DRAW);

	glEnableVertexAttribArray(depthWriteShader->attribute("voxentry"));
	glVertexAttribPointer(depthWriteShader->attribute("voxentry"), 3, GL_INT, GL_FALSE, sizeof(VoxelEntry), 0);
	//glEnableVertexAttribArray(depthWriteShader->attribute("SDFVolumeBasePtr_vert"));
	//glVertexAttribPointer(depthWriteShader->attribute("SDFVolumeBasePtr_vert"), 1, GL_INT, GL_FALSE, sizeof(VoxelEntry), BUFFER_OFFSET(sizeof(glm::ivec3)));

	//unbind
	glBindVertexArray(0);
	/*--------------------------------*/

	glGenBuffers(1, &numOccupiedBlocks_handle);
	glBindBuffer(GL_ARRAY_BUFFER, numOccupiedBlocks_handle);
	glBufferData(GL_ARRAY_BUFFER, sizeof(unsigned int), &numOccupiedBlocks, GL_STATIC_COPY);

	glGenBuffers(1, &SDF_VolumeBuffer_handle);
	glBindBuffer(GL_ARRAY_BUFFER, SDF_VolumeBuffer_handle);
	glBufferData(GL_ARRAY_BUFFER, sizeof(Voxel) * voxelBlockSize * voxelBlockSize * voxelBlockSize * numVoxelBlocks, nullptr, GL_STATIC_COPY);

	//unbind
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//glBindBuffer(GL_ARRAY_BUFFER, 0);
	//register with SDF_Hashtable class

}

//just to check if memory is mapped correctly. TODO : remove later
void SDFRenderer::printSDFdata() {
	//glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

	glBindBuffer(GL_ARRAY_BUFFER, numOccupiedBlocks_handle);
	glGetBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(unsigned int), &numOccupiedBlocks);
	std::vector<VoxelEntry> vox_entries;
	//now copy back VoxelEntries
	vox_entries.resize(numOccupiedBlocks);
	glBindBuffer(GL_ARRAY_BUFFER, compactHashTable_handle);
	glGetBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(VoxelEntry) * numOccupiedBlocks, vox_entries.data());

	std::vector<Voxel> sdf_chunks;
	//std::array<Voxel, numOccupiedBlocks*512> sdf_chunks;
	glBindBuffer(GL_ARRAY_BUFFER, SDF_VolumeBuffer_handle);
	sdf_chunks.resize(numOccupiedBlocks * 512);
	glGetBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(Voxel) * 512 * numOccupiedBlocks, sdf_chunks.data());
	glBindBuffer(GL_ARRAY_BUFFER, 0);


	std::ofstream fout("SDF_dump.txt");
	fout << std::fixed << std::showpoint;
	fout << std::setprecision(4);
	fout << std::setw(7);
	fout << "numOccupiedBlocks from GL :" << numOccupiedBlocks << "\n";
	//for (auto &i : vox_entries) {
	//	fout << "pos : (" << i.pos.x << ", " << i.pos.y << ", " << i.pos.z << ") ptr = " << i.ptr << " offset = " << i.offset << "\n";
	//}

	fout <<"\nSDFs \n\n";
	for(int i = 0; i < numOccupiedBlocks; ++i) {
		fout << i << ") : ";
		fout << "pos : (" << vox_entries[i].pos.x << ", " << vox_entries[i].pos.y << ", " << vox_entries[i].pos.z << ") ptr = " << vox_entries[i].ptr << " offset = " << vox_entries[i].offset << "\n";
		for (int j = 0; j < 512; ++j) {
			fout << sdf_chunks[i * 512 + j].sdf << "\t";
		}
		fout << "\n\n\n";
	}
	fout.close();

}

void SDFRenderer::drawSDF(ShaderProgram &shader, const glm::mat4& viewMat) {
	glBindVertexArray(SDF_VAO);
	//glBindBuffer(GL_ARRAY_BUFFER, compactHashTable_handle);
	//glVertexAttribPointer(raycast_shader->attribute("voxentry"), 3, GL_INT, GL_FALSE, sizeof(VoxelEntry), 0);
	shader.use();
	glEnableVertexAttribArray(shader.attribute("voxentry"));
	glUniformMatrix4fv(shader.uniform("VP"), 1, false, glm::value_ptr(viewMat));
	//glUniformMatrix4fv(raycast_shader->uniform("projMat"), 1, false, glm::value_ptr(projMat));
	glDrawArrays(GL_POINTS, 0, numOccupiedBlocks); //1);//
												   //glBindBuffer(GL_ARRAY_BUFFER, 0);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	glBindVertexArray(0);
}

void SDFRenderer::drawToFrontAndBack(const glm::mat4& viewMat) {
	//glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
	//First pass - render depth for front face
	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
	glFrontFace(GL_CW);	//IMPORTANT - Need to do this because we're looking along +Z axis
	fbo_front->enable();
	glClear(GL_DEPTH_BUFFER_BIT);
	glCullFace(GL_BACK);
	glDepthFunc(GL_LESS);
	depthWriteShader->use();
	glUniform1f(depthWriteShader->uniform("windowWidth"), windowWidth);
	glUniform1f(depthWriteShader->uniform("windowHeight"), windowHeight);
	glBindImageTexture(1, fbo_front->getSDFVolPtrTexID(), 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);
	drawSDF(*depthWriteShader, viewMat);
	fbo_front->disable();
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	//Second pass - render depth for front face

	fbo_back->enable();
	glClear(GL_DEPTH_BUFFER_BIT);
	glCullFace(GL_FRONT);
	glDepthFunc(GL_LESS);	//ideally should be GL_GREATER as per groundai article, but GL_LESS with custom depth-compare-shader works
	//raycast_shader->use();
	//drawSDF(*raycast_shader, viewMat);
	depthWriteShader->use();
	glUniform1f(depthWriteShader->uniform("windowWidth"), windowWidth);
	glUniform1f(depthWriteShader->uniform("windowHeight"), windowHeight);
	glBindImageTexture(1, fbo_back->getSDFVolPtrTexID(), 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8UI);
	//glActiveTexture(GL_TEXTURE0);
	//glBindTexture(GL_TEXTURE_2D, fbo_front->getDepthTexID());
	//glUniform1i(depthWriteShader->uniform("prevDepthTexture"), 0);
	drawSDF(*depthWriteShader, viewMat);
	fbo_back->disable();
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

	//raycast_shader->uniform("viewMat")

	//TODO - remove this later
	//raycast_shader->use();
	//drawSDF(*raycast_shader, viewMat);
}

void SDFRenderer::render(const glm::mat4& viewMat) {
	drawToFrontAndBack(viewMat);
	//draw to screen
}

SDFRenderer::~SDFRenderer() {

}
