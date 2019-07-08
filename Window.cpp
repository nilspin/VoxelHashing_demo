#include "Window.h"

void Window::swap()
{
	SDL_GL_SwapWindow(window);
}

Window::Window()
{
	Initialize();
}


Window::~Window()
{
	SDL_GL_DeleteContext(context);
	SDL_DestroyWindow(window);
}

bool Window::Initialize()
{
	start = 0;
	SDL_Init(SDL_INIT_EVERYTHING);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 16);
	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 0);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);

	window = SDL_CreateWindow("KinectVis", 0, 0, 1280, 960, SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL /*| SDL_WINDOW_RESIZABLE*/);
	context = SDL_GL_CreateContext(window);

	if (!gladLoadGL())
	{
		throw std::runtime_error("GLAD initialization failed");
	}

	glEnable(GL_DEPTH_TEST);

#ifdef NDEBUG
	//Enable Debug output
	glEnable(GL_DEBUG_OUTPUT);
	glDebugMessageCallback(MessageCallback, nullptr);
	glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
	glDebugMessageInsert(GL_DEBUG_SOURCE_APPLICATION, GL_DEBUG_TYPE_MARKER, 0,
		GL_DEBUG_SEVERITY_NOTIFICATION, -1, "Start debugging");
	return true;
#endif // DEBUG
}
