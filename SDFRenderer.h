#ifndef SDF_RENDERER_H
#define SDF_RENDERER_H

#include "prereq.h"

//class SDF_Hashtable;

class SDFRenderer {
	unsigned int numOccupiedBlocks = 0;
	GLuint SDF_VAO;
	GLuint numOccupiedBlocks_handle = -1;
	GLuint SDF_VolumeBuffer_handle;
	GLuint compactHashTable_handle;

	GLuint FBO;
	GLuint depthTexture_front;
	GLuint depthTexture_back;


public:
	friend class SDF_Hashtable;
	SDFRenderer();
	~SDFRenderer();
	void printSDFdata();
	void render(const glm::mat4&);
	//friend void registerGLtoCUDA(SDFRenderer*);
	//SDFRenderer(const SDFRenderer&) = delete;
	//SDFRenderer& operator=(const SDFRenderer&) = delete;
};
#endif // !SDF_RENDERER_H
