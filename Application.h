#ifndef APPLICATION_H
#define APPLICATION_H

#include "prereq.h"
#include "Window.h"
#include "ShaderProgram.hpp"
#include "camera.h"
#include "Frustum.h"
#include "CameraTracking.h"

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
  glm::mat4 deltaT = glm::mat4(1);  //we need to find this each iteration

  //Frustum
  Frustum frustum;

	//Shader
	unique_ptr<ShaderProgram> drawVertexMap;
	
  //CameraTracker
  unique_ptr<CameraTracking> tracker;
	
	//Texture & images
	GLuint depthTexture1, depthTexture2;
	uint16_t *image1=nullptr;
	uint16_t *image2=nullptr;
	
	//OpenGL Buffer objects
	GLuint inputVBO;
  GLuint inputNormalVBO;
	GLuint targetVBO;
  GLuint targetNormalVBO;
	GLuint inputVAO;
  GLuint targetVAO;


	void SetupShaders();
	void SetupBuffers();
  void processEvents();
  void draw(const glm::mat4&);

  //CUDA stuff
  //----for out incoming frame-----
  struct cudaGraphicsResource *cuda_input_resource;
  struct cudaGraphicsResource *cuda_inputNormals_resource;
    //----for frame to be compared against----
  struct cudaGraphicsResource *cuda_target_resource;
  struct cudaGraphicsResource *cuda_targetNormals_resource;


  uint16_t *d_depthInput, *d_depthTarget;
  glm::vec4* d_input;
  glm::vec4* d_inputNormals;
  glm::vec4* d_target;
  glm::vec4* d_targetNormals;

};

#endif //APPLICATION_H
