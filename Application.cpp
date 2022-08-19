#define STB_IMAGE_IMPLEMENTATION

#include "Application.h"
//#include "help"
#include<fstream>
#include<cuda_runtime_api.h>
//#include<cuda.h>

#include<glm/gtx/string_cast.hpp>
#include<cuda_gl_interop.h>
#include "termcolor.hpp"
#include "helper_cuda.h"
#include "stb_image.h"

SDL_Event event;
using glm::vec3;
using glm::vec4;
using glm::mat4;
//using namespace glm;

int tempFramesToIntegrate = 3;
static int initOnce           = 0;
//Takes device pointers, calculates correct position and normals
extern "C" void generatePositionAndNormals(float4 *positions, float4* normals, const std::uint16_t *depth);

//void SetupInputData(
Application::Application() {

	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	//Set up frustum
  frustum.setFromVectors(vec3(0,0,1), vec3(0,0,0), vec3(1,0,0), vec3(0,1,0), 0.1, 500.0, 45, 1.3333);
  //stbi_set_flip_vertically_on_load(true); //Keep commented for now

  //Start tracker
  tracker = make_unique<CameraTracking>(m_DepthWidth, m_DepthHeight);
  //Depth Fusion
  fusionModule = make_unique<SDF_Hashtable>();
  //Render to screen
  sdfRenderer = make_unique<SDFRenderer>();
  //register renderer to depthfusion class so CUDA can directly write to GL's buffers
  fusionModule->registerGLtoCUDA(*sdfRenderer);

  //put into cuda device buffer
  const int DEPTH_SIZE = sizeof(std::uint16_t)*m_DepthHeight*m_DepthWidth;
  checkCudaErrors(cudaMalloc((void**)&d_inputDepths, DEPTH_SIZE));
  checkCudaErrors(cudaMalloc((void**)&d_targetDepths, DEPTH_SIZE));
  checkCudaErrors(cudaDeviceSynchronize());

  cam.setPosition(glm::vec3(0, 0, 0));
  cam.setProjectionMatrix(proj);
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

  SetupShaders();
  SetupBuffers();

  //checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_targetVerts_resource, 0));
  checkCudaErrors(cudaDeviceSynchronize());

}

Application::~Application()
{
  glBindVertexArray(0);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGraphicsUnregisterResource(cuda_inputVerts_resource));
  checkCudaErrors(cudaGraphicsUnregisterResource(cuda_inputNormals_resource));
  checkCudaErrors(cudaGraphicsUnregisterResource(cuda_targetVerts_resource));
  checkCudaErrors(cudaGraphicsUnregisterResource(cuda_targetNormals_resource));

	if(h_tempDepthFrame){ //todo : refactor this later
		stbi_image_free(h_tempDepthFrame);
		image1 = nullptr;
	}
	if(image1){
		stbi_image_free(image1);
		image1 = nullptr;
	}
	if(image2){
		stbi_image_free(image2);
		image2 = nullptr;
	}
  checkCudaErrors(cudaFree(d_inputDepths));
  checkCudaErrors(cudaFree(d_targetDepths));
  glDeleteBuffers(1, &inputVBO);
  glDeleteBuffers(1, &targetVBO);
}

void Application::run()
{
	//setup date for first frame
	const std::string rootPath = std::string("assets/T");
	const std::string format 	 = std::string(".png");
	static int framesIntegrated = 0;
  static int        startFrameIdx    = 11;
	std::string inputFile1 = std::to_string(startFrameIdx);
	std::string inputPath1 = rootPath + inputFile1 + format;

	if(!image1) {
		image1 = stbi_load_16(inputPath1.c_str(), &m_DepthWidth, &m_DepthHeight, &m_colorChannels, 0);
		cout<<"Loaded image : "<<inputPath1<<"\n";
	}
	if(image1 == nullptr)
	{
		cout<<"could not read first image file!"<<endl; exit(0);
	}

	const int DEPTH_SIZE = sizeof(std::uint16_t)*m_DepthHeight*m_DepthWidth;
	checkCudaErrors(cudaMemcpy(d_inputDepths, image1, DEPTH_SIZE, cudaMemcpyHostToDevice));

	//Map GL resources to CUDA once for initial frame -- do it just for input
	size_t allocdDeviceBufSize;
	//input-verts
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_inputVerts_resource, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_inputVerts, &allocdDeviceBufSize, cuda_inputVerts_resource));
	checkCudaErrors(cudaMemset(d_inputVerts, 0, allocdDeviceBufSize));

	//input-normals
	checkCudaErrors(cudaGraphicsMapResources(1, &cuda_inputNormals_resource, 0));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_inputNormals, &allocdDeviceBufSize, cuda_inputNormals_resource));
	checkCudaErrors(cudaMemset(d_inputNormals, 0, allocdDeviceBufSize));

	//generate verts/normals from depth -- just for first input frame
	generatePositionAndNormals(d_inputVerts,  d_inputNormals,  d_inputDepths);

	//integrate the first frame at origin
	float4x4 identity;
	identity.setIdentity();
	fusionModule->integrate(identity, d_inputVerts, d_inputNormals);

	//sanity-check
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_inputVerts_resource, 0));
	checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_inputNormals_resource, 0));
	d_inputVerts = d_inputNormals = nullptr;
	framesIntegrated++;
  while (!quit)
  {
    processEvents();  //Event loop
    while (startFrameIdx + framesIntegrated <= startFrameIdx + 1)
    {
      std::string inputFile2 = std::to_string(startFrameIdx + framesIntegrated);
      std::string inputPath2 = rootPath + inputFile2 + format;
      // if(!image2) {
      image2 = stbi_load_16(inputPath2.c_str(), &m_DepthWidth, &m_DepthHeight, &m_colorChannels, 0);
      cout << "Loaded image : " << inputPath2 << "\n";
      //}
      if (image2 == nullptr)
      {
        cout << "could not read second image file!" << endl;
        exit(0);
      }

      // copy depth frame to GPU
      checkCudaErrors(cudaMemcpy(d_targetDepths, image2, DEPTH_SIZE, cudaMemcpyHostToDevice));
      checkCudaErrors(cudaDeviceSynchronize());

      stbi_image_free(image2);
      image2 = nullptr;

      // Map GL resources to CUDA -- TODO : do I need to do this each frame?
      size_t allocdDeviceBufSize_inputVerts;
      size_t allocdDeviceBufSize_inputNormals;
      size_t allocdDeviceBufSize_targetVerts;
      size_t allocdDeviceBufSize_targetNormals;
      // input-verts
      checkCudaErrors(cudaGraphicsMapResources(1, &cuda_inputVerts_resource, 0));
      checkCudaErrors(
          cudaGraphicsResourceGetMappedPointer((void**)&d_inputVerts, &allocdDeviceBufSize_inputVerts, cuda_inputVerts_resource));

      // input-normals
      checkCudaErrors(cudaGraphicsMapResources(1, &cuda_inputNormals_resource, 0));
      checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_inputNormals, &allocdDeviceBufSize_inputNormals,
                                                           cuda_inputNormals_resource));

      // target-verts
      checkCudaErrors(cudaGraphicsMapResources(1, &cuda_targetVerts_resource, 0));
      checkCudaErrors(
          cudaGraphicsResourceGetMappedPointer((void**)&d_targetVerts, &allocdDeviceBufSize_targetVerts, cuda_targetVerts_resource));

      // target-normals
      checkCudaErrors(cudaGraphicsMapResources(1, &cuda_targetNormals_resource, 0));
      checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_targetNormals, &allocdDeviceBufSize_targetNormals,
                                                           cuda_targetNormals_resource));

      // VBOs allocated
      std::cout << "\nAllocated input VBO size: " << allocdDeviceBufSize << "\n";
      generatePositionAndNormals(d_targetVerts, d_targetNormals, d_targetDepths);

      //<HACKY, NON_ELEGANT SHIT> -- allocate the image pyramid device buffers
      //(we don't do it before cus buffer size isn't known)
      if (initOnce == 0)
      {
        bool allocStatus = tracker->AllocImagePyramids(d_inputVerts, d_inputNormals, d_inputDepths, d_targetVerts, d_targetNormals,
                                                       d_targetDepths);
        initOnce         = 1;
      }
      //</HACKY, NON_ELEGANT SHIT>

      tracker->Align(d_inputVerts, d_inputNormals, d_targetVerts, d_targetNormals, d_inputDepths, d_targetDepths);
      globalT = glm::make_mat4(tracker->getGlobalTransform().data());
      deltaT  = glm::make_mat4(tracker->getDeltaTransform().data());

      localTransforms.push_back(deltaT);
      globalTransforms.push_back(globalT);
      // globalT = glm::transpose(globalT);
      std::cout << termcolor::on_blue << "Final global transform : \n" << termcolor::reset << glm::to_string(globalT) << "\n";
      std::cout << termcolor::on_yellow << "Final delta  transform : \n" << termcolor::reset << glm::to_string(deltaT) << "\n";

      // Depth integration into volume
      auto transMatGlob  = tracker->getGlobalTransform();
      auto transMatLocal = tracker->getDeltaTransform();
      // transMat = transMat.inverse().eval();
      float4x4 global_transform = float4x4(transMatGlob.data());
      float4x4 local_transform  = float4x4(transMatLocal.data());
      // float4x4 global_transform = float4x4(tracker->getGlobalTransform().data());
      global_transform.transpose();

      if ((framesIntegrated % (tempFramesToIntegrate))==0) //Todo : temp code. remove later
      { 
				fusionModule->integrate(global_transform, d_targetVerts, d_targetNormals);
			}

			//sdfRenderer->printSDFdata();
   /**/
			{
				//Cleanup
				//checkCudaErrors(cudaMemset(d_inputVerts, 0, allocdDeviceBufSize_inputVerts));
				checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_inputVerts_resource, 0));

				//checkCudaErrors(cudaMemset(d_inputNormals, 0, allocdDeviceBufSize_inputNormals));
				checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_inputNormals_resource, 0));

				//checkCudaErrors(cudaMemset(d_targetVerts, 0, allocdDeviceBufSize_targetVerts));
				checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_targetVerts_resource, 0));

				//checkCudaErrors(cudaMemset(d_targetNormals, 0, allocdDeviceBufSize_targetNormals));
				checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_targetNormals_resource, 0));

				checkCudaErrors(cudaDeviceSynchronize());
			}

			//now make the target frames as new input
			auto temp = image1;
			image1 = image2;
			image2 = temp;

			float4 *temp_f4 = d_inputVerts;
			d_inputVerts = d_targetVerts;
			d_targetVerts = temp_f4;

			temp_f4 = d_inputNormals;
			d_inputNormals = d_targetNormals;
			d_targetNormals = temp_f4;

			tracker->swapBuffers(); //sets current target as new input

			framesIntegrated++;

		}


		///------RENDERING-OUTPUT-------///
    //First things first
    cam.calcMatrices();
		vec3 camPos = cam.getCamPos();
    GLfloat time = SDL_GetTicks();
    view = cam.getViewMatrix();
		//scaling : convert voxelblock dims to world-space dims
		mat4 model = glm::scale(glm::mat4(1.0f), glm::vec3(voxelSize*voxelBlockSize));
		mat4 VP = proj*view;	// *model;
		mat4 MV = view * model;
		mat4 MVP = proj*view*model;

		//render TSDFs
		//glDisable(GL_DEPTH_TEST);
		//glDisable(GL_CULL_FACE);
		//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		//sdfRenderer->render(MV, proj, camPos); //MV, P

		//or render input point-cloud
		//glBindFramebuffer(GL_FRAMEBUFFER, 0);
		//glDepthFunc(GL_TRUE);
		glDisable(GL_DEPTH_TEST);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		draw(VP);

		//dump SDF data
		//sdfRenderer->printSDFdata();

    //Draw frustum
    //frustum.draw(VP);

    window.swap();
    //quit=true;
  }
}

void Application::draw(const glm::mat4& vp)
{
  //glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  drawVertexMap->use();
  glm::mat4 MVP = vp*glm::mat4(1);
  glm::mat4 newMVP = vp*deltaT;
  glm::vec3 camPos = cam.getCamPos();

  //checkCudaErrors(cudaGraphicsMapResources(1, &cuda_inputVerts_resource, 0));
  glUniform3f(drawVertexMap->uniform("CamPos"), camPos.x, camPos.y, camPos.z);
  glUniformMatrix4fv(drawVertexMap->uniform("MVP"), 1, false, glm::value_ptr(newMVP));
  glUniform3f(drawVertexMap->uniform("LightPos"), 0.0f, 0.0f, 0.0f);


  glBindVertexArray(inputVAO);
  glUniform3f(drawVertexMap->uniform("ShadeColor"), 0.258, 0.956, 0.560); //old
  //glUniform3f(drawVertexMap->uniform("ShadeColor"), 0.4335, 0.8358, 0.7774);
  glDrawArrays(GL_POINTS, 0, 640*480);


  glBindVertexArray(targetVAO);
  glUniformMatrix4fv(drawVertexMap->uniform("MVP"), 1, false, glm::value_ptr(MVP));
  glUniform3f(drawVertexMap->uniform("ShadeColor"), 0.956, 0.721, 0.254); //old
  //glUniform3f(drawVertexMap->uniform("ShadeColor"), 0.6562, 0.1718, 0.2851);
  glDrawArrays(GL_POINTS, 0, 640*480);
  glBindVertexArray(0);


  //TESTING---------------
  // std::vector<glm::vec4> outputGL(20);
  // glBindBuffer(GL_ARRAY_BUFFER, inputVBO);
  // glGetBufferSubData(GL_ARRAY_BUFFER, 0, 20*sizeof(glm::vec4), outputGL.data());
  // glBindBuffer(GL_ARRAY_BUFFER, 0);
  // std::cout<<"\nCHECKING: GL output: "<<" "<<outputGL[0].x<<" "<<outputGL[0].y<<" "<<outputGL[0].z<<" "<<outputGL[0].w<<"\n";
  //----------------------
}

void Application::SetupShaders() {
  drawVertexMap = make_unique<ShaderProgram>();
  drawVertexMap->initFromFiles("shaders/MainShader.vert",
                               /*"shaders/MainShader.geom",*/
                               "shaders/MainShader.frag");
  drawVertexMap->addAttribute("positions");
  drawVertexMap->addAttribute("normals");
  drawVertexMap->addUniform("MVP");
  drawVertexMap->addUniform("ShadeColor");
  drawVertexMap->addUniform("LightPos");
  drawVertexMap->addUniform("CamPos");
}


void Application::SetupBuffers() {

  const int ARRAY_SIZE = m_DepthWidth * m_DepthHeight * sizeof(glm::vec4);

  //-------------INPUT BUFFER------------------------------
  glGenVertexArrays(1, &inputVAO);
  glBindVertexArray(inputVAO);
  //As we go along register buffers with CUDA as well
  glGenBuffers(1, &inputVBO);
  glBindBuffer(GL_ARRAY_BUFFER, inputVBO);
  glBufferData(GL_ARRAY_BUFFER, ARRAY_SIZE, nullptr, GL_STATIC_DRAW);
  glVertexAttribPointer(drawVertexMap->attribute("positions"), 4, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(drawVertexMap->attribute("positions"));

  glGenBuffers(1, &inputNormalVBO);
  glBindBuffer(GL_ARRAY_BUFFER, inputNormalVBO);
  glBufferData(GL_ARRAY_BUFFER, ARRAY_SIZE, nullptr, GL_STATIC_DRAW);
  glVertexAttribPointer(drawVertexMap->attribute("normals"), 4, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(drawVertexMap->attribute("normals"));

  //Unbind and do CUDA stuff
  glBindVertexArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  //-------------TARGET-BUFFER------------------------
  glGenVertexArrays(1, &targetVAO);
  glBindVertexArray(targetVAO);

  glGenBuffers(1, &targetVBO);
  glBindBuffer(GL_ARRAY_BUFFER, targetVBO);
  glBufferData(GL_ARRAY_BUFFER, ARRAY_SIZE, nullptr, GL_STATIC_DRAW);
  glVertexAttribPointer(drawVertexMap->attribute("positions"), 4, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(drawVertexMap->attribute("positions"));

  glGenBuffers(1, &targetNormalVBO);
  glBindBuffer(GL_ARRAY_BUFFER, targetNormalVBO);
  glBufferData(GL_ARRAY_BUFFER, ARRAY_SIZE, nullptr, GL_STATIC_DRAW);
  glVertexAttribPointer(drawVertexMap->attribute("normals"), 4, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(drawVertexMap->attribute("normals"));

  //Unbind and do CUDA stuff
  glBindVertexArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  //--------Register with CUDA-----
  checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_inputVerts_resource, inputVBO, cudaGraphicsRegisterFlagsNone));
  checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_inputNormals_resource, inputNormalVBO, cudaGraphicsRegisterFlagsNone));
  checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_targetVerts_resource, targetVBO, cudaGraphicsRegisterFlagsNone));
  checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_targetNormals_resource, targetNormalVBO, cudaGraphicsRegisterFlagsNone));

}

void Application::processEvents() {
  while (SDL_PollEvent(&event) != 0)
    {
      switch (event.type)
      {
        case SDL_QUIT:	//if X windowkey is pressed then quit
          quit = true;

        case SDL_KEYDOWN:	//if ESC is pressed then quit

          switch (event.key.keysym.sym)
          {
			case SDLK_p:
			  //sdfRenderer->printDebugImage();
			  break;

            case SDLK_q:
              quit = true;
              break;

            case SDLK_w:
              cam.move(FORWARD);
              break;

            case SDLK_s:
              cam.move(BACK);
              break;

            case SDLK_a:
              cam.move(LEFT);
              break;

            case SDLK_d:
              cam.move(RIGHT);
              break;

            case SDLK_UP:
              cam.move(UP);
              break;

            case SDLK_DOWN:
              cam.move(DOWN);
              break;

            case SDLK_LEFT:
              cam.move(ROT_LEFT);
              break;

            case SDLK_RIGHT:
              cam.move(ROT_RIGHT);
              break;


            case SDLK_r:
              cam.reset();
              std::cout << "Reset button pressed \n";
              break;

          }
          break;

        case SDL_MOUSEMOTION:
          cam.rotate();
          break;
      }
    }
}
