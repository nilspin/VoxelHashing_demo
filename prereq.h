#ifndef PREREQ_H
#define PREREQ_H

//std stuff
#include<iostream>
#include<cstring>
#include<string>
#include<sstream>
#include<vector>
#include<fstream>
#include<memory>

//OpenGL loader on Windows
#include "opengl_win.h"

//SDL
#include <SDL.h>
#include <SDL_opengl.h>

//GLM stuff
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_CUDA
#define GLM_SWIZZLE
#include<glm/glm.hpp>
#include<glm/gtc/matrix_transform.hpp>
#include<glm/gtc/type_ptr.hpp>
#include<glm/gtx/transform.hpp>

extern SDL_Event event;
#endif //PREREQ_H

