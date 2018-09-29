#define STB_IMAGE_IMPLEMENTATION

#include "Application.h"
//#include "help"
#include<fstream>
#include<cuda_runtime_api.h>
//#include<cuda.h>

#include<cuda_gl_interop.h>
#include "helper_cuda.h"
#include "stb_image.h"

SDL_Event event;
using glm::vec3;
using glm::vec4;
using glm::mat4;
//using namespace glm;

Application::Application() {
  frustum.setFromVectors(vec3(0,0,-1), vec3(0,0,0), vec3(1,0,0), vec3(0,1,0), 5.0, 700.0, 45, 1.3333);
  image1 = stbi_load_16("assets/T0.png", &DepthWidth, &DepthHeight, &channels, 0);
  image2 = stbi_load_16("assets/T5.png", &DepthWidth, &DepthHeight, &channels, 0);
  if(image1 == nullptr) {cout<<"could not read first image file!"<<endl; exit(0);}
  if(image2 == nullptr) {cout<<"could not read second image file!"<<endl; exit(0);}
  tracker = unique_ptr<CameraTracking>(new CameraTracking(DepthWidth, DepthHeight));

  //put into cuda device buffer
  const int DEPTH_SIZE = sizeof(uint16_t)*DepthHeight*DepthWidth;
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
  checkCudaErrors(cudaGraphicsMapResources(1, &cuda_input_resource, 0));
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_input, &returnedBufferSize, cuda_input_resource));
  checkCudaErrors(cudaMemset(d_input, 0, returnedBufferSize));

  checkCudaErrors(cudaGraphicsMapResources(1, &cuda_inputNormals_resource, 0));
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_inputNormals, &returnedBufferSize, cuda_inputNormals_resource));
  checkCudaErrors(cudaMemset(d_inputNormals, 0, returnedBufferSize));

  checkCudaErrors(cudaGraphicsMapResources(1, &cuda_target_resource, 0));
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_target, &returnedBufferSize, cuda_target_resource));
  checkCudaErrors(cudaMemset(d_input, 0, returnedBufferSize));

  checkCudaErrors(cudaGraphicsMapResources(1, &cuda_targetNormals_resource, 0));
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_targetNormals, &returnedBufferSize, cuda_targetNormals_resource));
  checkCudaErrors(cudaMemset(d_input, 0, returnedBufferSize));

  std::cout<<"\nAllocated input VBO size: "<<returnedBufferSize<<"\n";
  tracker->Align(d_input, d_inputNormals, d_target, d_targetNormals, d_depthInput, d_depthTarget);
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
    GLfloat time = SDL_GetTicks();
    view = cam.getViewMatrix();
    MVP = proj*view;//*model;

    //tracker.Align(d_input, d_inputNormals, d_target, d_targetNormals, d_depthInput, d_depthTarget);
    //checkCudaErrors(cudaDeviceSynchronize());
    draw(MVP);
    
    mat4 scaleMat =  glm::scale(vec3(1000));
    mat4 newMVP = proj*view;//*scaleMat
    //Draw frustum
    //frustum.draw(newMVP);

    window.swap();
    //quit=true;
  }
}

void Application::draw(const glm::mat4& mvp)
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  drawVertexMap->use();
  
  //checkCudaErrors(cudaGraphicsMapResources(1, &cuda_input_resource, 0));
  glUniformMatrix4fv(drawVertexMap->uniform("MVP"), 1, false, glm::value_ptr(mvp));
  
  
  glBindVertexArray(inputVAO);
  glUniform3f(drawVertexMap->uniform("shadeColor"), 0.258, 0.956, 0.560);
  glDrawArrays(GL_POINTS, 0, 640*480);
  

  glBindVertexArray(targetVAO);
  glUniform3f(drawVertexMap->uniform("shadeColor"), 0.956, 0.721, 0.254);
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
                               "shaders/MainShader.geom",
                               "shaders/MainShader.frag");
  drawVertexMap->addAttribute("positions");
  drawVertexMap->addAttribute("normals");
  drawVertexMap->addUniform("MVP");
  drawVertexMap->addUniform("shadeColor");
}


void Application::SetupBuffers() {
  
  const int ARRAY_SIZE = DepthWidth * DepthHeight * sizeof(glm::vec4);
  
  //-------------INPUT BUFFER------------------------------
  glGenVertexArrays(1, &inputVAO);
  glBindVertexArray(inputVAO);
  //As we go along register buffers with CUDA as well
  glGenBuffers(1, &inputVBO);
  glBindBuffer(GL_ARRAY_BUFFER, inputVBO);
  glBufferData(GL_ARRAY_BUFFER, ARRAY_SIZE, nullptr, GL_DYNAMIC_DRAW);
  glVertexAttribPointer(drawVertexMap->attribute("positions"), 4, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(drawVertexMap->attribute("positions"));

  glGenBuffers(1, &inputNormalVBO);
  glBindBuffer(GL_ARRAY_BUFFER, inputNormalVBO);
  glBufferData(GL_ARRAY_BUFFER, ARRAY_SIZE, nullptr, GL_DYNAMIC_DRAW);
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
  glBufferData(GL_ARRAY_BUFFER, ARRAY_SIZE, nullptr, GL_DYNAMIC_DRAW);
  glVertexAttribPointer(drawVertexMap->attribute("positions"), 4, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(drawVertexMap->attribute("positions"));

  glGenBuffers(1, &targetNormalVBO);
  glBindBuffer(GL_ARRAY_BUFFER, targetNormalVBO);
  glBufferData(GL_ARRAY_BUFFER, ARRAY_SIZE, nullptr, GL_DYNAMIC_DRAW);
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
