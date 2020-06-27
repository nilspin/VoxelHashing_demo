#ifndef SDFRENDERER_UTILS_H
#define SDFRENDERER_UTILS_H

#include<memory>
#include<iostream>
#include "FBO.hpp"
#include "ShaderProgram.hpp"
#include "VoxelDataStructures.h"

std::unique_ptr<FBO> setupFBO_w_intTex();
//std::unique_ptr<ShaderProgram> setupRaycastShader();
std::unique_ptr<ShaderProgram> setupDepthWriteShader();
std::unique_ptr<ShaderProgram> setupInstancedCubeDrawShader();
//std::unique_ptr<ShaderProgram> setupDrawLinearDepthShader();
//void setupSDF_GL_buffer(GLuint SDF_VAO,...);//
void generateCanvas(GLuint&, GLuint&, GLuint&);
void generateUnitCube(GLuint&);
GLuint setup_Debug_SSBO();

#endif //SDFRENDERER_UTILS_H
