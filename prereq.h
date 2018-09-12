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


//GL
#include<GL/glew.h>
#include<GL/glu.h>
#include<GL/gl.h>

//SDL
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

//GLM stuff
#define GLM_ENABLE_EXPERIMENTAL
#include<glm/glm.hpp>
#include<glm/gtc/matrix_transform.hpp>
#include<glm/gtc/type_ptr.hpp>
#include<glm/gtx/transform.hpp>

extern SDL_Event event;
#endif //PREREQ_H

