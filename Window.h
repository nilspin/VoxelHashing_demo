#pragma once
#include "prereq.h"

class Window
{
public:
	void swap();
	Window();
	~Window();
private:
	SDL_GLContext context;
	uint32_t start = 0;
	SDL_Window* window = NULL;
	bool Initialize();
};

