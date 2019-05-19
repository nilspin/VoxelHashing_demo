/*
camera.h
OpenGL Camera Code
*/
#ifndef CAMERA_H
#define CAMERA_H

#include "prereq.h"

/*
#include <glm.hpp>
#include <gtc/matrix_transform.hpp>
#include <SDL.h>
*/

//extern SDL_Window* window;
//SDL_Event event;

using namespace std;

enum CameraDirection {
	LEFT, RIGHT, FORWARD, BACK, UP, DOWN, ROT_LEFT, ROT_RIGHT
};

class Camera {
private :

	glm::mat4 ViewMatrix = glm::lookAt(position, glm::vec3(0,0,1), up);
	// Projection matrix : 45° Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
	glm::mat4 ProjectionMatrix = glm::perspective(initialFoV, 4.0f / 3.0f, 0.1f, 100.0f);

	// Initial position : on +Z
	glm::vec3 position,InitialPos;
	// Initial horizontal angle : toward -Z
	float horizontalAngle = 3.14f;//0.0f;//
	// Initial vertical angle : none
	float verticalAngle = 3.14f;//0.0f;//
	// Initial Field of View
	float initialFoV = 45.0f;
	// Actual direction
	glm::vec3 direction;
	// Right vector
	glm::vec3 right;
	// Up
	glm::vec3 up = glm::vec3(0,1,0);
	// LookAt
//	glm::vec3 lookAt = glm::vec3(0, 0, -10);
	// Time difference
	float timeDifference = 0;

	float speed = 0.01f; // 3 units / second
	float mouseSpeed = 0.01f;

	int temp = 0;

public:
//	Camera();
	//Camera(SDL_Event& e) {event=e;}
//	~Camera();
	glm::vec3 getUpDir(){ return up; }
	glm::vec3 getRightDir(){ return right; }
	glm::vec3 getCamPos(){ return position; }
	glm::vec3 getDirection(){ return direction; }	//return current direction we're looking at

	void setViewMat(glm::mat4 mat){ ViewMatrix = mat; }
	void setPosition(glm::vec3);
	void move(CameraDirection);
	void computeMatricesFromInputs();
	void setProjectionMatrix(glm::mat4);
	glm::mat4 getViewMatrix();
	glm::mat4 getProjectionMatrix();
	void reset();
	void rotate();
	void calcMatrices();
	void s(int i){ temp += i; }

};
#endif
