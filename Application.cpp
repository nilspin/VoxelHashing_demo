#define STB_IMAGE_IMPLEMENTATION

#include "cudaHelper.h"
#include "Application.h"
#include "stb_image.h"
#include<fstream>
#include<cuda_runtime_api.h>
#include<cuda.h>
#include<cuda_gl_interop.h>

SDL_Event event;

Application::Application() {
  frustum.setFromVectors(vec3(0,0,-1), vec3(0,0,0), vec3(1,0,0), vec3(0,1,0), 5.0, 700.0, 45, 1.3333);
  image1 = stbi_load("assets/T0.png", &DepthWidth, &DepthHeight, &channels, 2);
  image2 = stbi_load("assets/T5.png", &DepthWidth, &DepthHeight, &channels, 2);
  if(image1 == nullptr) {cout<<"could not read image file!"<<endl; exit(0);}

  //put into cuda device buffer
  const int DEPTH_SIZE = sizeof(uint16_t)*DepthHeight*DepthWidth;
  checkCudaErrors(cudaMalloc((void**)&d_depthInput, DEPTH_SIZE));
  checkCudaErrors(cudaMalloc((void**)&d_depthTarget, DEPTH_SIZE));
  checkCudaErrors(cudaMemcpy(d_depthInput, image1, DEPTH_SIZE, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_depthTarget, image2, DEPTH_SIZE, cudaMemcpyHostToDevice));
  stbi_image_free(image1);
  stbi_image_free(image2);


  cam.setPosition(glm::vec3(0, 0, 0));
  cam.setProjectionMatrix(proj);
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  SetupShaders();
  SetupBuffers();
}

Application::~Application() {
  checkCudaErrors(cudaGraphicsUnregisterResource(cuda_input_resource));
  checkCudaErrors(cudaGraphicsUnregisterResource(cuda_target_resource));
  checkCudaErrors(cudaFree(d_depthInput));
  checkCudaErrors(cudaFree(d_depthTarget));
}

void Application::run() {
  while (!quit)
  {
    processEvents();  //Event loop

    //First things first
    cam.calcMatrices();
    GLfloat time = SDL_GetTicks();
    view = cam.getViewMatrix();
    MVP = proj*view*model;

    tracker.Align(d_input, d_inputNormals, d_target, d_targetNormals, d_depthInput, d_depthTarget);
    draw(MVP);
    
    mat4 scaleMat =  glm::scale(vec3(1000));
    mat4 newMVP = proj*view;//*scaleMat
    //Draw frustum
    frustum.draw(MVP);

    window.swap();
  }
}

void Application::draw(const glm::mat4& mvp)
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  drawVertexMap->use();
  glUniformMatrix4fv(drawVertexMap->uniform("MVP"), 1, false, glm::value_ptr(mvp));

  glBindVertexArray(vertexArray);
  glUniform3f(drawVertexMap->uniform("shadeColor"), 1, 0, 0);
  glDrawArrays(GL_POINTS, 0, 640*480);

  glUniform3f(drawVertexMap->uniform("shadeColor"), 0, 1, 0);
  glDrawArrays(GL_POINTS, 0, 640*480);
}

void Application::SetupShaders() {
  drawVertexMap = unique_ptr<ShaderProgram>(new ShaderProgram());
  drawVertexMap->initFromFiles("shaders/MainShader.vert", "shaders/MainShader.frag");
  drawVertexMap->addAttribute("positions");
  drawVertexMap->addUniform("MVP");
  drawVertexMap->addUniform("shadeColor");
}


void Application::SetupBuffers() {
  
  //Create Vertex Array Object
  glGenVertexArrays(1, &vertexArray);
  glBindVertexArray(vertexArray);
  const uint ARRAY_SIZE = DepthWidth * DepthHeight * sizeof(glm::vec4);
  //As we go along register buffers with CUDA as well
  glGenBuffers(1, &inputVBO);
  glBindBuffer(GL_ARRAY_BUFFER, inputVBO);
  glBufferData(GL_ARRAY_BUFFER, ARRAY_SIZE, nullptr, GL_DYNAMIC_DRAW);
  checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_input_resource, inputVBO, cudaGraphicsRegisterFlagsNone));
  checkCudaErrors(cudaGraphicsMapResources(1, &cuda_input_resource, 0));
  size_t returnedBufferSize;
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_input, &returnedBufferSize, cuda_input_resource));
  std::cout<<"Allocated input VBO size: "<<returnedBufferSize<<"\n";

  glGenBuffers(1, &targetVBO);
  glBindBuffer(GL_ARRAY_BUFFER, targetVBO);
  glBufferData(GL_ARRAY_BUFFER, ARRAY_SIZE, nullptr, GL_DYNAMIC_DRAW);
  checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_target_resource, targetVBO, cudaGraphicsRegisterFlagsNone));
  checkCudaErrors(cudaGraphicsMapResources(1, &cuda_target_resource, 0));
  checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&d_target, &returnedBufferSize, cuda_target_resource));
  std::cout<<"Allocated target VBO size: "<<returnedBufferSize<<"\n";

  //Now set up VBOs
  checkCudaErrors(cudaMalloc((void**)&d_inputNormals, ARRAY_SIZE));
  checkCudaErrors(cudaMalloc((void**)&d_targetNormals, ARRAY_SIZE));
  checkCudaErrors(cudaMalloc((void**)&d_correspondence, ARRAY_SIZE));
  //Assign attribs
  glEnableVertexAttribArray(drawVertexMap->attribute("positions"));
  glVertexAttribPointer(drawVertexMap->attribute("positions"), 4, GL_FLOAT, GL_FALSE, 0, 0);
  //glBindVertexArray(0);	//unbind VAO
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