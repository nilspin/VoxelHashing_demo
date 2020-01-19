#ifndef SDFRENDERER_UTILS_H
#define SDFRENDERER_UTILS_H

#include<memory>
#include<iostream>
#include "FBO.hpp"
#include "ShaderProgram.hpp"

std::unique_ptr<FBO> setupFBO_w_intTex();
std::unique_ptr<ShaderProgram> setupRaycastShader();
std::unique_ptr<ShaderProgram> setupDepthWriteShader();
std::unique_ptr<ShaderProgram> setupDrawLinearDepthShader();

#endif //SDFRENDERER_UTILS_H
