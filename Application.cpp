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

//Takes device pointers, calculates correct position and normals
extern "C" void preProcess(float4 *positions, float4* normals, const std::uint16_t *depth);

//void SetupInputData(
Application::Application() {
	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	//Set up frustum
  frustum.setFromVectors(vec3(0,0,1), vec3(0,0,0), vec3(1,0,0), vec3(0,1,0), 0.1, 500.0, 45, 1.3333);
  //stbi_set_flip_vertically_on_load(true); //Keep commented for now
  image1 = stbi_load_16("assets/T0.png", &DepthWidth, &DepthHeight, &channels, 0);
  image2 = stbi_load_16("assets/T1.png", &DepthWidth, &DepthHeight, &channels, 0);
  if(image1 == nullptr) {cout<<"could not read first image file!"<<endl; exit(0);}
  if(image2 == nullptr) {cout<<"could not read second image file!"<<endl; exit(0);}
  //Start tracker
  tracker = unique_ptr<CameraTracking>(new CameraTracking(DepthWidth, DepthHeight));
  //Depth Fusion
  fusionModule = unique_ptr<SDF_Hashtable>(new SDF_Hashtable());
  //Render to screen
  sdfRenderer = unique_ptr<SDFRenderer>(new SDFRenderer());
  //register renderer to depthfusion class so CUDA can directly write to GL's buffers
  fusionModule->registerGLtoCUDA(*sdfRenderer);

  //put into cuda device buffer
  const int DEPTH_SIZE = sizeof(std::uint16_t)*DepthHeight*DepthWidth;
  checkCudaErrors(cudaMalloc((void**)&d_depthInput, DEPTH_SIZE));
  checkCudaErrors(cudaMalloc((void**)&d_depthTarget, DEPTH_SIZE));
  checkCudaErrors(cudaMemcpy(d_depthInput, image1, DEPTH_SIZE, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_depthTarget, image2, DEPTH_SIZE, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaDeviceSynchronize());
  stbi_image_free(image1);
  stbi_image_free(image2);


  cam.setPosition(glm::vec3(0, 0, 0));
  cam.setProjectionMatrix(proj);
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  SetupShaders();
  SetupBuffers();

  //TODO: Move into gameloop
  size_t returnedBufferSize;
  //input-verts
  checkCudaErrors(cudaGraphicsMapResources(1, &cuda_input_resource, 0));
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_input, &returnedBufferSize, cuda_input_resource));
  checkCudaErrors(cudaMemset(d_input, 0, returnedBufferSize));

  //input-normals
  checkCudaErrors(cudaGraphicsMapResources(1, &cuda_inputNormals_resource, 0));
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_inputNormals, &returnedBufferSize, cuda_inputNormals_resource));
  checkCudaErrors(cudaMemset(d_inputNormals, 0, returnedBufferSize));

  //target-verts
  checkCudaErrors(cudaGraphicsMapResources(1, &cuda_target_resource, 0));
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_target, &returnedBufferSize, cuda_target_resource));
  checkCudaErrors(cudaMemset(d_input, 0, returnedBufferSize));

  //target-normals
  checkCudaErrors(cudaGraphicsMapResources(1, &cuda_targetNormals_resource, 0));
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_targetNormals, &returnedBufferSize, cuda_targetNormals_resource));
  checkCudaErrors(cudaMemset(d_input, 0, returnedBufferSize));

  //VBOs allocated
  std::cout<<"\nAllocated input VBO size: "<<returnedBufferSize<<"\n";
  preProcess(d_input, d_inputNormals, d_depthInput);
  preProcess(d_target, d_targetNormals, d_depthTarget);
  tracker->Align(d_input, d_inputNormals, d_target, d_targetNormals, d_depthInput, d_depthTarget);
  deltaT = glm::make_mat4(tracker->getTransform().data());
  //deltaT = glm::transpose(deltaT);
  std::cout << termcolor::on_blue<< "Final rigid transform : \n" << termcolor::reset<< glm::to_string(deltaT) << "\n";

  //TODO Depth integration into volume
  float4x4 global_transform = float4x4(tracker->getTransform().data());
  float4x4 identity;
  identity.setIdentity();
  fusionModule->integrate(identity, d_input, d_inputNormals);
  //sdfRenderer->printSDFdata();
  //fusionModule->integrate(global_transform, d_target, d_targetNormals);
  checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_input_resource, 0));
  checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_inputNormals_resource, 0));
  checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_target_resource, 0));
  checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_targetNormals_resource, 0));

  //*Testing output
  //   std::vector<glm::vec4> outputCUDA(640*480);
  //   checkCudaErrors(cudaMemcpy(outputCUDA.data(), d_input, 640*480*sizeof(glm::vec4), cudaMemcpyDeviceToHost));
  //   std::ofstream fout("correspondenceData.txt");
  //   std::for_each(outputCUDA.begin(), outputCUDA.end(), [&fout](const glm::vec4 &n){fout<<n.x<<" "<<n.y<<" "<<n.z<<" "<<n.w<<"\n";});
  //std::cout<<"\nCHECKING: CUDA output: "<<" "<<outputCUDA[0].x<<" "<<outputCUDA[0].y<<" "<<outputCUDA[0].z<<" "<<outputCUDA[0].w<<"\n";
  //*/

  //checkCudaErrors(cudaGraphicsUnmapResources(1, &cuda_target_resource, 0));
  checkCudaErrors(cudaDeviceSynchronize());

}

Application::~Application() {

  glBindVertexArray(0);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGraphicsUnregisterResource(cuda_input_resource));
  checkCudaErrors(cudaGraphicsUnregisterResource(cuda_inputNormals_resource));
  checkCudaErrors(cudaGraphicsUnregisterResource(cuda_target_resource));
  checkCudaErrors(cudaGraphicsUnregisterResource(cuda_targetNormals_resource));

  checkCudaErrors(cudaFree(d_depthInput));
  checkCudaErrors(cudaFree(d_depthTarget));
  glDeleteBuffers(1, &inputVBO);
  glDeleteBuffers(1, &targetVBO);
}

void Application::run() {
  while (!quit)
  {
    processEvents();  //Event loop

    //First things first
    cam.calcMatrices();
	vec3 camPos = cam.getCamPos();
    GLfloat time = SDL_GetTicks();
    view = cam.getViewMatrix();
	//scaling : convert voxelblock dims to world-space dims
	mat4 model = glm::scale(glm::mat4(1.0f), glm::vec3(voxelSize*voxelBlockSize)); //glm::mat4(1.0f);// 
	mat4 VP = proj*view;	// *model;
	mat4 MV = view * model;
	mat4 MVP = proj*view*model;
	//std::cout<<"\n"<<glm::to_string(MVP)<<"\n\n";
    //tracker.Align(d_input, d_inputNormals, d_target, d_targetNormals, d_depthInput, d_depthTarget);
    //checkCudaErrors(cudaDeviceSynchronize());
	glEnable(GL_DEPTH_TEST);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//sdfRenderer->render(MVP); //MV, P
	sdfRenderer->render(MV, proj, camPos); //MV, P
	//glBindFramebuffer(GL_FRAMEBUFFER, 0);
	//glDepthFunc(GL_TRUE);
	//glDisable(GL_DEPTH_TEST);
	//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//draw(VP);
	//sdfRenderer->printSDFdata();


    //mat4 scaleMat =  glm::scale(vec3(1000));
    //mat4 newMVP = proj*view;//*scaleMat
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

  //checkCudaErrors(cudaGraphicsMapResources(1, &cuda_input_resource, 0));
  glUniform3f(drawVertexMap->uniform("CamPos"), camPos.x, camPos.y, camPos.z);
  glUniformMatrix4fv(drawVertexMap->uniform("MVP"), 1, false, glm::value_ptr(newMVP));
  glUniform3f(drawVertexMap->uniform("LightPos"), 0.0f, 0.0f, 0.0f);


  glBindVertexArray(inputVAO);
  glUniform3f(drawVertexMap->uniform("ShadeColor"), 0.258, 0.956, 0.560);
  glDrawArrays(GL_POINTS, 0, 640*480);


  glBindVertexArray(targetVAO);
  glUniformMatrix4fv(drawVertexMap->uniform("MVP"), 1, false, glm::value_ptr(MVP));
  glUniform3f(drawVertexMap->uniform("ShadeColor"), 0.956, 0.721, 0.254);
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
  drawVertexMap = unique_ptr<ShaderProgram>(new ShaderProgram());
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

  const int ARRAY_SIZE = DepthWidth * DepthHeight * sizeof(glm::vec4);

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
  checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_input_resource, inputVBO, cudaGraphicsRegisterFlagsNone));
  checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_inputNormals_resource, inputNormalVBO, cudaGraphicsRegisterFlagsNone));
  checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_target_resource, targetVBO, cudaGraphicsRegisterFlagsNone));
  checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_targetNormals_resource, targetNormalVBO, cudaGraphicsRegisterFlagsNone));

  //-------------Now allocate rest of CUDA arrays for which we don't need GL----------------
  //checkCudaErrors(cudaMalloc((void**)&d_inputNormals, ARRAY_SIZE));
  //checkCudaErrors(cudaMalloc((void**)&d_targetNormals, ARRAY_SIZE));

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
