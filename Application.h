#ifndef APPLICATION_H
#define APPLICATION_H

#include "prereq.h"
#include "Window.h"
#include "ShaderProgram.hpp"
#include "camera.h"
#include "Frustum.h"

using namespace std;
using namespace glm;

#define BUFFER_OFFSET(i) ((char*)NULL + (i))

class Application
{
public:
	void run();
	Application();
	~Application();
private:
	//Dims
	int DepthWidth = 0;
	int DepthHeight = 0;
	int channels = 0;
	float wide = 0.1;

	//Window and events
	Window window;
	//SDL_Event event;
	
	bool quit=false;

	//Camera & transforms
	Camera cam;
	glm::mat4 model = glm::mat4(1);
	glm::mat4 view = glm::mat4(1);
	glm::mat4 proj = glm::perspective(45.0f, 1.3333f, 0.1f, 5000.0f);
	glm::mat4 MVP = glm::mat4(1);

    //Frustum
    Frustum frustum;

	//Shader
	unique_ptr<ShaderProgram> drawVertexMap;
	
	
	//Texture & images
	GLuint depthTexture1, depthTexture2;
	uint8_t *image1=nullptr;
	uint8_t *image2=nullptr;
	
	//OpenGL Buffer objects
	vector<glm::ivec2>	texCoords;
	GLuint vertexBuffer;
	GLuint texCoordBuffer;
	GLuint vertexArray;


	void UploadDepthToTexture(uint8_t*, int, int);
	void SetupShaders();
	void SetupBuffers();
	void SetupDepthTextures();
  void processEvents();

};

#endif //APPLICATION_H
