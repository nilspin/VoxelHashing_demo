#define STB_IMAGE_IMPLEMENTATION

#include "Application.h"
#include "stb_image.h"

extern SDL_Event event;

Application::Application() {
	//cam(event);
	positionBuffer.resize(640*480);
	texCoords.resize(640*480);
	image1 = stbi_load("assets/depth1.png", &DepthWidth, &DepthHeight, &channels, 0);
	image2 = stbi_load("assets/depth2.png", &DepthWidth, &DepthHeight, &channels, 0);
	if(image1 == nullptr) {cout<<"could not read file!"<<endl;}
	cam.SetPosition(glm::vec3(0,0,10));
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
					cam.Reset();
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

		glUniformMatrix4fv(drawVertexMap->uniform("MVP"), 1, false, glm::value_ptr(MVP));

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glBindVertexArray(vertexArray);
		drawVertexMap->use();
		//inputSource.UploadDepthToTexture();

		glUniform1i(drawVertexMap->uniform("depthTexture"), 0);
		glDrawArrays(GL_POINTS, 0, 640*480);
		window.swap();
	}
}

void Application::SetupShaders() {
	drawVertexMap = (make_unique<ShaderProgram>());
	drawVertexMap->initFromFiles("shaders/MainShader.vert", "shaders/MainShader.frag");
	drawVertexMap->addAttribute("position");
	drawVertexMap->addAttribute("texCoords");
	drawVertexMap->addUniform("depthTexture");
	drawVertexMap->addUniform("MVP");
}

void Application::SetupDepthTextures() {
	glGenTextures(1, &depthTexture1);
	glBindTexture(GL_TEXTURE_2D, depthTexture1);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, DepthWidth, DepthHeight, 0, GL_RED, GL_FLOAT, 0);
	//filtering
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	//Texture2 
	glGenTextures(1, &depthTexture2);
	glBindTexture(GL_TEXTURE_2D, depthTexture2);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, DepthWidth, DepthHeight, 0, GL_RED, GL_FLOAT, 0);
	//filtering
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
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
			positionBuffer[i * DepthHeight + j] = (glm::vec2(((GLfloat)i*wide), ((GLfloat)j*wide)));
		}
	}
	//Create Vertex Array Object
	glGenVertexArrays(1, &vertexArray);
	glBindVertexArray(vertexArray);

	//VBOs
	glGenBuffers(1, &vertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, (DepthWidth * DepthHeight) * sizeof(glm::vec2), positionBuffer.data(), GL_STATIC_DRAW);
	//Assign attribs
	glEnableVertexAttribArray(drawVertexMap->attribute("position"));
	glVertexAttribPointer(drawVertexMap->attribute("position"), 2, GL_FLOAT, GL_FALSE, 0, 0);
	
	glGenBuffers(1, &texCoordBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, texCoordBuffer);
	glBufferData(GL_ARRAY_BUFFER, DepthWidth * DepthHeight * sizeof(glm::ivec2), texCoords.data(), GL_STATIC_DRAW);
	//Assign attribs
	glEnableVertexAttribArray(drawVertexMap->attribute("texCoords"));
	glVertexAttribPointer(drawVertexMap->attribute("texCoords"), 2, GL_FLOAT, GL_FALSE, 0, 0);
	glBindVertexArray(0);	//unbind VAO
}

void Application::UploadDepthToTexture(uint8_t* image, int texID, int texUnit) {
	uint16_t *buffer = reinterpret_cast<uint16_t*>(image);//((uint16_t)image[0] << 8) | (uint16_t)image[1];//image;
	//HRESULT getInternalBuffer = IR_Frame->AccessUnderlyingBuffer(&bufferSize, &buffer);
	const uint16_t* pBufferEnd = buffer + (DepthWidth * DepthHeight);
	uint8_t* dataArrayPointer = tempDataArray;

	while (buffer < pBufferEnd)
	{
		uint16_t depth = *buffer;
		//BYTE intensity = static_cast<BYTE>((depth >= 500) && (depth <= 65535) ? (depth % 256) : 0);
		char lo = depth & 0xFF;
		char hi = depth >> 8;
		*dataArrayPointer = hi;
		*(dataArrayPointer + 1) = lo;
		*(dataArrayPointer + 2) = 0;

		dataArrayPointer = (dataArrayPointer + 3);
		++buffer;
		
	}
//	const void* dataArrayPointer = tempDataArray;
	//now write this to a texture for sanity 
	//					int result = stbi_write_bmp("depthmap.png", DepthWidth, DepthHeight, 3, dataArrayPointer);
	//LOAD TEXTURE HERE
	glActiveTexture(GL_TEXTURE0 + texUnit);
	glBindTexture(GL_TEXTURE_2D, texID);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, DepthWidth, DepthHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, tempDataArray/*dataArrayPointer*/);

}

Application::~Application() {
	stbi_image_free(image1);
	stbi_image_free(image2);
}
