#include "Application.h"
#include <memory>

int main(int argc, char *argv[])
{

	//std::unique_ptr<Application> app = std::make_unique<Application>();
	std::unique_ptr<Application> app(make_unique<Application>());
	app->run();
	return 0;
}