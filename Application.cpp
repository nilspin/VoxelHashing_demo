#define STB_IMAGE_IMPLEMENTATION


#include "Application.h"
#include "stb_image.h"
#include<fstream>

SDL_Event event;

Application::Application() {
  frustum.setFromVectors(vec3(0,0,-1), vec3(0,0,0), vec3(1,0,0), vec3(0,1,0), 5.0, 700.0, 45, 1.3333);
  texCoords.resize(640*480);
  image1 = stbi_load("assets/d1.png", &DepthWidth, &DepthHeight, &channels, 2);
  image2 = stbi_load("assets/d2.png", &DepthWidth, &DepthHeight, &channels, 2);
  if(image1 == nullptr) {cout<<"could not read image file!"<<endl; exit(0);}
  cam.setPosition(glm::vec3(0, 0, 0));
  cam.setProjectionMatrix(proj);
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
  SetupShaders();
  SetupBuffers();
  SetupDepthTextures();
  UploadDepthToTexture(image1, depthTexture1, 0);
  UploadDepthToTexture(image2, depthTexture2, 1);
}

void Application::run() {
  while (!quit)
  {
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

    //First things first
    cam.calcMatrices();
    GLfloat time = SDL_GetTicks();
    view = cam.getViewMatrix();
    MVP = proj*view*model;

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    drawVertexMap->use();

    glUniformMatrix4fv(drawVertexMap->uniform("MVP"), 1, false, glm::value_ptr(MVP));

    glBindVertexArray(vertexArray);
    //inputSource.UploadDepthToTexture();


    //first depthMap
    glUniform1i(drawVertexMap->uniform("depthTexture"), 0);
    glUniform3f(drawVertexMap->uniform("shadeColor"), 1, 0, 0);
    glDrawArrays(GL_POINTS, 0, 640*480);
    //second depthMap
    glUniform1i(drawVertexMap->uniform("depthTexture"), 1);
    glUniform3f(drawVertexMap->uniform("shadeColor"), 0, 1, 0);
    glDrawArrays(GL_POINTS, 0, 640*480);
    //
    mat4 scaleMat =  glm::scale(vec3(1000));
    mat4 newMVP = proj*view;//*scaleMat
    //Draw frustum
    frustum.draw(MVP);

    window.swap();
  }
}

void Application::SetupShaders() {
  drawVertexMap = (make_unique<ShaderProgram>());
  drawVertexMap->initFromFiles("shaders/MainShader.vert", "shaders/MainShader.frag");
  drawVertexMap->addAttribute("texCoords");
  drawVertexMap->addUniform("depthTexture");
  drawVertexMap->addUniform("MVP");
  drawVertexMap->addUniform("shadeColor");
}

void Application::SetupDepthTextures() {
  glGenTextures(1, &depthTexture1);
  glBindTexture(GL_TEXTURE_2D, depthTexture1);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, DepthWidth, DepthHeight, 0, GL_RG, GL_UNSIGNED_BYTE, 0);
  //filtering
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  //Texture2
  glGenTextures(1, &depthTexture2);
  glBindTexture(GL_TEXTURE_2D, depthTexture2);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, DepthWidth, DepthHeight, 0, GL_RG, GL_UNSIGNED_BYTE, 0);
  //filtering
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glBindTexture(GL_TEXTURE_2D, 0);
}


void Application::SetupBuffers() {
  //First create array on RAM
  for (int i = 0; i < DepthWidth; ++i)
  {
    for (int j = 0; j < DepthHeight; ++j)
    {
      texCoords[i * DepthHeight + j] = (glm::vec2(i, j));
    }
  }
  //Create Vertex Array Object
  glGenVertexArrays(1, &vertexArray);
  glBindVertexArray(vertexArray);

  glGenBuffers(1, &texCoordBuffer);
  glBindBuffer(GL_ARRAY_BUFFER, texCoordBuffer);
  glBufferData(GL_ARRAY_BUFFER, DepthWidth * DepthHeight * sizeof(glm::ivec2), texCoords.data(), GL_STATIC_DRAW);
  //Assign attribs
  glEnableVertexAttribArray(drawVertexMap->attribute("texCoords"));
  glVertexAttribPointer(drawVertexMap->attribute("texCoords"), 2, GL_FLOAT, GL_FALSE, 0, 0);
  glBindVertexArray(0);	//unbind VAO
}

void Application::UploadDepthToTexture(uint8_t* image, int texID, int texUnit) {

  glActiveTexture(GL_TEXTURE0 + texUnit);
  glBindTexture(GL_TEXTURE_2D, texID);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, 640, 480, 0, GL_RG, GL_UNSIGNED_BYTE, image);

}

Application::~Application() {
  stbi_image_free(image1);
  stbi_image_free(image2);
}
