#ifndef SDF_RENDERER_H
#define SDF_RENDERER_H

#include "prereq.h"
#include "common.h"
#include "ShaderProgram.hpp"
#include "FBO.hpp"

//class SDF_Hashtable;

class SDFRenderer {
	unsigned int numOccupiedBlocks = 0;
	GLuint SDF_VAO;
	GLuint CanvasVAO, CanvasVBO;
	GLuint numOccupiedBlocks_handle = -1;
	GLuint SDF_VolumeBuffer_handle;
	GLuint compactHashTable_handle;
	std::unique_ptr<FBO> fbo_front;
	std::unique_ptr<FBO> fbo_back;

	//std::unique_ptr<ShaderProgram> raycast_shader;
	std::unique_ptr<ShaderProgram> depthWriteShader;	//Shader to compare depths between fbo_front and fbo_back FBOs
	//std::unique_ptr<ShaderProgram> drawLinearDepth;	//I forgot what this was!
	const float zNear = 0.1f;
	const float zFar = 5.0f;
	glm::mat4 projMat = glm::perspective(45.0f, 1.3333f, zNear, zFar);
	//---------------DEBUG--------------------------------
	GLuint dbg_R16I_imgTex;	//for front face
	GLuint debug_ssbo;	//debug ssbo to keep track of per-pixel metrics
	/*such as
	  * ID of ray hitting the front buffer
	  * start position of ray hitting the block (world space)
	  *
	  * ID of ray hitting the back buffer
	  * stop position of ray hitting the block (world space)
	  */
	//std::unique_ptr<ShaderProgram> debugSDFInfo;	//debug shader that writes what's
	//supposed to be drawn into an SSBO
	//----------------------------------------------------

public:
	friend class SDF_Hashtable;
	SDFRenderer();
	~SDFRenderer();
	void CreateImageBuffer();
	void printSDFdata();
	void render(const glm::mat4&);
	void drawToFrontAndBack(const glm::mat4&);
	void drawSDF(ShaderProgram &, const glm::mat4&);
	//friend void registerGLtoCUDA(SDFRenderer*);
	//SDFRenderer(const SDFRenderer&) = delete;
	//SDFRenderer& operator=(const SDFRenderer&) = delete;
};
#endif // !SDF_RENDERER_H
