#include "camera.h"

glm::mat4 Camera::getViewMatrix(){
	return ViewMatrix;
}
glm::mat4 Camera::getProjectionMatrix(){
	return ProjectionMatrix;
}

void Camera::rotate()
{
	static double lastTime = SDL_GetTicks();
	double currentTime = SDL_GetTicks();
	float deltaTime = float(currentTime - lastTime);
	timeDifference = deltaTime;
	// Get mouse position
	int xpos = event.motion.xrel;
	int ypos = event.motion.yrel;

	// Compute new orientation
	horizontalAngle += mouseSpeed * float(/*(1024 / 2) -*/ -xpos);
	verticalAngle += mouseSpeed * float(/*(768 / 2) -*/ -ypos);

	// For the next frame, the "last time" will be "now"
	lastTime = currentTime;

}

void Camera::move(CameraDirection dir)
{
	switch (dir)
	{
	case FORWARD:	// Move forward
		position += direction * speed;
		break;

	case BACK:		// Move backward
		position -= direction * speed;
		break;

	case RIGHT:		// Move right
		position += right * speed;
		break;

	case LEFT:		// Move left
		position -= right * speed;
		break;

	case UP:		// Look up
		verticalAngle += 0.1;
		break;

	case DOWN:
		verticalAngle -= 0.1;
		break;

	case ROT_LEFT:
		horizontalAngle += 0.1;
		break;


	case ROT_RIGHT:
		horizontalAngle -= 0.1;
		break;


	}

}

void Camera::setPosition(glm::vec3 pos)
{
	position = InitialPos = pos;
}

void Camera::setProjectionMatrix(glm::mat4 proj)
{
	ProjectionMatrix = proj;
}
void Camera::calcMatrices()
{
	// Direction : Spherical coordinates to Cartesian coordinates conversion
	direction = glm::vec3(
		cos(verticalAngle) * sin(horizontalAngle),
		sin(verticalAngle),
		cos(verticalAngle) * cos(horizontalAngle)
		);

	// Right vector
	right = glm::vec3(
		sin(horizontalAngle - 3.14f / 2.0f),
		0,
		cos(horizontalAngle - 3.14f / 2.0f)
		);

	// Up vector
	up = glm::cross(right, direction);

	float FoV = initialFoV;

	// Camera matrix
	ViewMatrix = glm::lookAt(
		position,           // Camera is here
		position + direction, // and looks here : at the same position, plus "direction"
		up                  // Head is up (set to 0,-1,0 to look upside-down)
		);
}

void Camera::reset()
{
	position = InitialPos;
	horizontalAngle = 3.14f;
	verticalAngle = 0.0f;
	calcMatrices();
}