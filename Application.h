#ifndef APPLICATION_H
#define APPLICATION_H

#include "prereq.h"
#include "Window.h"
#include "ShaderProgram.hpp"
#include "camera.h"
#include "Frustum.h"
#include "CameraTracking.h"
#include "SDFRenderer.h"
#include "SDF_Hashtable.h"
#include "helper_cuda.h"

#include <thrust/device_vector.h>

#define BUFFER_OFFSET(i) ((char*)NULL + (i))

using uint16_t = std::uint16_t;

class Application
{
public:
	void run();
	Application();
	~Application();
private:
	//Dims
	int m_DepthWidth = 640;
	int m_DepthHeight = 480;
	int m_colorChannels = 1;
	float wide = 0.1;

	//Window and events
	Window window;
	//SDL_Event event;

	bool quit=false;

	//Camera & transforms
	Camera cam;
	glm::mat4 model = glm::mat4(1);
	glm::mat4 view = glm::mat4(1);
	glm::mat4 proj = glm::perspective(45.0f, 1.3333f, 0.01f, 5.0f);
	glm::mat4 MVP = glm::mat4(1);
	glm::mat4 deltaT  = glm::mat4(1);  //transform from prev frame to latest
	glm::mat4 globalT = glm::mat4(1);  //global frame transformation

	vector<glm::mat4> localTransforms;
	vector<glm::mat4> globalTransforms;

	//Frustum
	Frustum frustum;

	//Shader
	unique_ptr<ShaderProgram> drawVertexMap;

	//CameraTracker
	unique_ptr<CameraTracking> tracker;

  	//GPU-hashtable
	unique_ptr<SDF_Hashtable> fusionModule;

	//Main renderer
	unique_ptr<SDFRenderer> sdfRenderer;

	//Texture & images
	GLuint depthTexture1, depthTexture2;
	std::uint16_t *image1	  				= nullptr;
	std::uint16_t *image2		  			= nullptr;
	std::uint16_t *h_tempDepthFrame  = nullptr;

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
  struct cudaGraphicsResource *cuda_inputVerts_resource;
  struct cudaGraphicsResource *cuda_inputNormals_resource;
    //----for frame to be compared against----
  struct cudaGraphicsResource *cuda_targetVerts_resource;
  struct cudaGraphicsResource *cuda_targetNormals_resource;


  std::uint16_t *d_inputDepths      = nullptr;
	std::uint16_t *d_targetDepths     = nullptr;
	std::uint16_t *d_tempDepths = nullptr;

	//store per-pixel (viewframe coordinates)+(normals) here
	float4* d_inputVerts    = nullptr;
  float4* d_inputNormals  = nullptr;
  float4* d_targetVerts   = nullptr;
  float4* d_targetNormals = nullptr;
  float4* d_tempVerts = nullptr;
  float4* d_tempNormals = nullptr;

};

#endif //APPLICATION_H
