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
#if defined(_WIN32)
  #include "opengl_win.h"
#elif defined(__linux__)
  #include "opengl_linux.h"
#endif

//SDL
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

//Make Eigen use Intel's MKL
//#define EIGEN_USE_MKL_ALL

//GLM stuff
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_SWIZZLE
#include<glm/glm.hpp>
#include<glm/gtc/matrix_transform.hpp>
#include<glm/gtc/type_ptr.hpp>
#include<glm/gtx/transform.hpp>

extern SDL_Event event;
#endif //PREREQ_H

