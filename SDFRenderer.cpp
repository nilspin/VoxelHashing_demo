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

	raycast_shader = setupRaycastShader();

	depthWriteShader = setupDepthWriteShader();

	drawLinearDepth = setupDrawLinearDepthShader();
	//generateCanvas();
	//raycast_shader->addUniform("projMat");

	/*----------VAO-------------------*/
	//init GL resources
	glGenVertexArrays(1, &SDF_VAO);
	glBindVertexArray(SDF_VAO);

	glGenBuffers(1, &compactHashTable_handle);
	glBindBuffer(GL_ARRAY_BUFFER, compactHashTable_handle);
	glBufferData(GL_ARRAY_BUFFER, sizeof(VoxelEntry) * numBuckets * bucketSize, nullptr, GL_STATIC_DRAW);
	glEnableVertexAttribArray(raycast_shader->attribute("voxentry"));
	glVertexAttribPointer(raycast_shader->attribute("voxentry"), 3, GL_INT, GL_FALSE, sizeof(VoxelEntry), 0);

	glEnableVertexAttribArray(drawLinearDepth->attribute("voxentry"));
	glVertexAttribPointer(drawLinearDepth->attribute("voxentry"), 3, GL_INT, GL_FALSE, sizeof(VoxelEntry), 0);
	//glEnableVertexAttribArray(drawLinearDepth->attribute("SDFVolumeBasePtr_vert"));
	//glVertexAttribPointer(drawLinearDepth->attribute("SDFVolumeBasePtr_vert"), 1, GL_INT, GL_FALSE, sizeof(VoxelEntry), BUFFER_OFFSET(sizeof(glm::ivec3)));


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

/*
//void SDFRenderer::drawSDF(ShaderProgram &shader, const glm::mat4& viewMat) {
//	glBindVertexArray(SDF_VAO);
//	//glBindBuffer(GL_ARRAY_BUFFER, compactHashTable_handle);
//	//glVertexAttribPointer(raycast_shader->attribute("voxentry"), 3, GL_INT, GL_FALSE, sizeof(VoxelEntry), 0);
//	raycast_shader->use();
//	glEnableVertexAttribArray(raycast_shader->attribute("voxentry"));
//	glUniformMatrix4fv(raycast_shader->uniform("VP"), 1, false, glm::value_ptr(viewMat));
//	//glUniformMatrix4fv(raycast_shader->uniform("projMat"), 1, false, glm::value_ptr(projMat));
//	glDrawArrays(GL_POINTS, 0, numOccupiedBlocks); //1);//
//	//glBindBuffer(GL_ARRAY_BUFFER, 0);
//	glBindVertexArray(0);
//}
*/

void SDFRenderer::CreateImageBuffer()	{
	std::vector<uint8_t> emptyImage(windowWidth * windowHeight * 4);
	std::fill(emptyImage.begin(), emptyImage.end(), 0);
	glGenTextures(GL_TEXTURE_2D, &dbg_R16I_imgTex);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8UI, windowWidth, windowHeight);
	glBindTexture(GL_TEXTURE_2D, 0);
}

void SDFRenderer::printDebugImage()	{
	std::vector<uint8_t> imageVector(windowWidth * windowHeight * 4);
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
	glBindTexture(GL_TEXTURE_2D, fbo_front->getSDFVolPtrTexID());
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, &imageVector[0]);
	glBindTexture(GL_TEXTURE_2D, 0);
	//stbi_flip_vertically_on_write(1);
	stbi_write_png("textureDump.png", windowWidth, windowHeight, 4, imageVector.data(), windowWidth * 4/*stride*/);
	std::cout<<"Screenshot dumped to file. \n";
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
	glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
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
	drawSDF(*raycast_shader, viewMat);
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

	/*
TODO : THIS HAS GOTTEN TOO MESSY!!! CLEAN THIS UP SOON

	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glDisable(GL_CULL_FACE);
	//glDisable(GL_DEPTH_TEST);
	//glDisable(GL_BLEND);
	//glDepthFunc(GL_LESS);
	//glBindVertexArray(CanvasVAO);
	drawLinearDepth->use();
	glUniformMatrix4fv(drawLinearDepth->uniform("VP"), 1, false, glm::value_ptr(viewMat));
	glUniformMatrix4fv(drawLinearDepth->uniform("invVP"), 1, false, glm::value_ptr(glm::inverse(viewMat)));
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, fbo_front->getDepthTexID());
	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_2D, fbo_back->getDepthTexID());
	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_2D, fbo_back->getSDFVolPtrTexID());
	glUniform1i(drawLinearDepth->uniform("startDepthTex"), 1);
	glUniform1i(drawLinearDepth->uniform("endDepthTex"), 2);
	glUniform1f(drawLinearDepth->uniform("windowWidth"), windowWidth);
	glUniform1f(drawLinearDepth->uniform("windowHeight"), windowHeight);
	//bind same compactifiedHashtable buffer as SSBO
	//GLuint voxelentry_ssbo_index = 0;
	//voxelentry_ssbo_index = glGetProgramResourceIndex(drawLinearDepth->getProgramHandle(), GL_SHADER_STORAGE_BLOCK, "VoxelEntry");
	//glShaderStorageBlockBinding(drawLinearDepth->getProgramHandle(), voxelentry_ssbo_index, 2);
	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, compactHashTable_handle);

	//bind sdf_voxels buffer as SSBO
	GLuint sdfvoxels_ssbo_index = 0;
	sdfvoxels_ssbo_index = glGetProgramResourceIndex(drawLinearDepth->getProgramHandle(), GL_SHADER_STORAGE_BLOCK, "SDFVolume");
	//glShaderStorageBlockBinding(drawLinearDepth->getProgramHandle(), sdfvoxels_ssbo_index, 3);
	glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, SDF_VolumeBuffer_handle);
	drawSDF(*drawLinearDepth, viewMat);
	//glDrawArrays(GL_TRIANGLES, 0, 6);
	glBindTexture(GL_TEXTURE_2D, 0);
	glBindVertexArray(0);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
*/
}

void SDFRenderer::generateCanvas() {
	GLfloat canvas[] = {		//DATA
		-1.0f,-1.0f,
		1.0f, -1.0f,
		-1.0f, 1.0f,

		1.0f, -1.0f,
		-1.0f, 1.0f,
		1.0f, 1.0f,
	};	//Don't need index data for this peasant mesh!

	glGenVertexArrays(1, &CanvasVAO);
	glBindVertexArray(CanvasVAO);
	glGenBuffers(1, &CanvasVBO);
	glBindBuffer(GL_ARRAY_BUFFER, CanvasVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(canvas), &canvas, GL_STATIC_DRAW);
	glVertexAttribPointer(drawLinearDepth->attribute("position"), 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(drawLinearDepth->attribute("position"));
	glBindVertexArray(0);	//unbind VAO
}

SDFRenderer::~SDFRenderer() {

}
