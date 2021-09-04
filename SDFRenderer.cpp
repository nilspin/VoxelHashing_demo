//#define STB_IMAGE_WRITE_IMPLEMENTATION

//#include "stb_image_write.h"
#include "common.h"
#include "SDFRenderer.h"
#include "SDFRendererUtils.h"
#include "VoxelDataStructures.h"
#include <vector>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cstddef>

#define BUFFER_OFFSET(i) ((char*)NULL + (i))

SDFRenderer::SDFRenderer() {

	fbo_front = setupFBO_w_intTex();

	fbo_back = setupFBO_w_intTex();

	//raycast_shader = setupRaycastShader();

	tempPassthroughShader = setupDepthWriteShader(); //TODO : cleanup

	instancedCubeDrawShader = setupInstancedCubeDrawShader();
	generateCanvas(CanvasVAO, CanvasVertVBO, CanvasTexCoordsVBO);
	//---------------Debug---------------
	//debug_ssbo = setup_Debug_SSBO();
	//-----------------------------------

	/*----------Scene VAO-------------------*/
	//first generate Box
	generateUnitCube(InstanceCubeVBO);

	//init rest of the GL resources required to render scene
	glGenVertexArrays(1, &Scene);
		glBindVertexArray(Scene);

		//attrib0 - boxVerts
		glBindBuffer(GL_ARRAY_BUFFER, InstanceCubeVBO); //no need for bufferdata, we've already set it up
			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
			glVertexAttribDivisor(0, 0);

		//attrib1 - boxCenters
		glGenBuffers(1, &compactHashTable_handle);
		glBindBuffer(GL_ARRAY_BUFFER, compactHashTable_handle);
			glEnableVertexAttribArray(1);
			glBufferData(GL_ARRAY_BUFFER, sizeof(VoxelEntry) * numBuckets * bucketSize, nullptr, GL_DYNAMIC_DRAW);
			glVertexAttribIPointer(1, 3, GL_INT, sizeof(VoxelEntry), nullptr); //boxCenters
			glVertexAttribDivisor(1, 1);
			//attrib2 - PtrId
			glEnableVertexAttribArray(2);
			glVertexAttribIPointer(2, 1, GL_INT, sizeof(VoxelEntry), reinterpret_cast<void *>(offsetof(VoxelEntry, ptr))); //PtrID
			glVertexAttribDivisor(2, 1);
			std::cout << "Sizeof(VoxelEntry) = " << sizeof(VoxelEntry) << "\n";

	//unbind
	glBindVertexArray(0);

	/*--------------------------------*/

	glGenBuffers(1, &numOccupiedBlocks_handle);
	glBindBuffer(GL_ARRAY_BUFFER, numOccupiedBlocks_handle);
	glBufferData(GL_ARRAY_BUFFER, sizeof(unsigned int), &numOccupiedBlocks, GL_DYNAMIC_COPY);

	//TODO : eventually make this an SSBO, not a regular vertex buffer!
	glGenBuffers(1, &SDF_VolumeBuffer_handle);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, SDF_VolumeBuffer_handle);
	glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(Voxel) * voxelBlockSize * voxelBlockSize * voxelBlockSize * numVoxelBlocks, nullptr, GL_DYNAMIC_COPY);

	//unbind
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
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

//void SDFRenderer::drawSDF(ShaderProgram &shader, const glm::mat4& viewMat) {
//	glBindVertexArray(Scene);
//	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, debug_ssbo);
//	shader.use();
//	glEnableVertexAttribArray(0);
//	glEnableVertexAttribArray(1);
//	glUniformMatrix4fv(shader.uniform("VP"), 1, false, glm::value_ptr(viewMat));
//	//glUniformMatrix4fv(raycast_shader->uniform("projMat"), 1, false, glm::value_ptr(projMat));
//	glDrawArrays(GL_POINTS, 0, numOccupiedBlocks); //1);//
//	glBindVertexArray(0);
//}

void SDFRenderer::drawToFrontAndBack(const glm::mat4& viewMat) {
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	//First pass - render depth for front face
	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
	glFrontFace(GL_CCW);	//IMPORTANT - Need to do this because we're looking along +Z axis

	fbo_back->enable();

	//clear before writing anything
	glClearTexImage(fbo_back->getSDFVolPtrTexID(), 0, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);

	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
	glCullFace(GL_BACK);
	//glDepthFunc(GL_LESS);
	glBindVertexArray(Scene);
	glEnableVertexAttribArray(0);	//boxVerts
	glEnableVertexAttribArray(1);	//boxCenters
	glEnableVertexAttribArray(2);	//VoxelBlockIDs
	instancedCubeDrawShader->use();
	//TODO : attach debug_ssbo here
	glUniformMatrix4fv(instancedCubeDrawShader->uniform("MVP"), 1, false, glm::value_ptr(viewMat));
	//glDrawArraysInstanced(GL_TRIANGLES, 0, 36,  numOccupiedBlocks);
	glDrawArraysInstanced(GL_TRIANGLES, 0, 36,  1);
	glBindVertexArray(0);

	fbo_back->disable();

	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

	//---------------Second pass - render depth for front face--------------


	fbo_front->enable();

	//clear before writing anything
	//TODO : clear all 4 channels
	glClearTexImage(fbo_front->getSDFVolPtrTexID(), 0, GL_RED_INTEGER, GL_UNSIGNED_INT, nullptr);

	//glEnable(GL_DEPTH_TEST);
	//glDepthFunc(GL_GREATER);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
	glCullFace(GL_FRONT);
	//glDepthFunc(GL_LESS);	//ideally should be GL_GREATER as per groundai article, but GL_LESS with custom depth-compare-shader works

	glBindVertexArray(Scene);
	glEnableVertexAttribArray(0);	//boxVerts
	glEnableVertexAttribArray(1);	//boxCenters
	glEnableVertexAttribArray(2);	//VoxelBlockIDs
	instancedCubeDrawShader->use();
	//TODO : attach debug_ssbo here
	glUniformMatrix4fv(instancedCubeDrawShader->uniform("MVP"), 1, false, glm::value_ptr(viewMat));
	//glDrawArraysInstanced(GL_TRIANGLES, 0, 36, numOccupiedBlocks);
	glDrawArraysInstanced(GL_TRIANGLES, 0, 36,  1);
	glBindVertexArray(0);

	fbo_front->disable();

	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

}

void SDFRenderer::render(const glm::mat4& MV, const glm::mat4& P, const glm::vec3& camPos) {
	glBindBuffer(GL_ARRAY_BUFFER, numOccupiedBlocks_handle);
	glGetBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(unsigned int), &numOccupiedBlocks);
	glm::mat4 MVP = P * MV;
	drawToFrontAndBack(MVP);
	//draw to screen

	//glFrontFace(GL_CW);	//IMPORTANT - Need to do this because we're looking along +Z axis
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glDisable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);
	tempPassthroughShader->use();
	glBindVertexArray(CanvasVAO);
	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, fbo_front->getSDFVolPtrTexID());
	glUniform1i(tempPassthroughShader->uniform("VoxelID_tex"), 0);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, fbo_front->getRayhitTexID());
	glUniform1i(tempPassthroughShader->uniform("rayHit_start"), 1);
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, fbo_back->getRayhitTexID());
	glUniform1i(tempPassthroughShader->uniform("rayHit_end"), 2);
	//glUniformMatrix4fv(tempPassthroughShader->uniform("invMVP"), 1, false, glm::value_ptr(glm::inverse(MVP)));
	//glUniform3fv(tempPassthroughShader->uniform("camPos"), 1, glm::value_ptr(camPos));

	//TODO: bind sdf_voxels buffer as SSBO
	///GLuint sdfvoxels_ssbo_index = 0;
	sdfvoxels_ssbo_index = glGetProgramResourceIndex(tempPassthroughShader->getProgramHandle(), GL_SHADER_STORAGE_BLOCK, "SDFVolume"); //Get block index
	glShaderStorageBlockBinding(tempPassthroughShader->getProgramHandle(), sdfvoxels_ssbo_index, 3); //connect shader storage block to ssbo
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, SDF_VolumeBuffer_handle);

	//Below two lines not needed
	//glUniformMatrix4fv(tempPassthroughShader->uniform("invModelViewMat"), 1, false, glm::value_ptr(glm::inverse(MV)));
	//glUniformMatrix4fv(tempPassthroughShader->uniform("invProjMat"), 1, false, glm::value_ptr(glm::inverse(P)));

	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);
	glActiveTexture(GL_TEXTURE0);  glBindTexture(GL_TEXTURE_2D, 0);
	glActiveTexture(GL_TEXTURE1);  glBindTexture(GL_TEXTURE_2D, 0);
	glActiveTexture(GL_TEXTURE2);  glBindTexture(GL_TEXTURE_2D, 0);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

SDFRenderer::~SDFRenderer() {

}
