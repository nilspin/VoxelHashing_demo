#ifndef SDF_RENDERER_H
#define SDF_RENDERER_H

#include "prereq.h"
#include "common.h"
#include "ShaderProgram.hpp"
#include "FBO.hpp"

//class SDF_Hashtable;

class SDFRenderer {
	unsigned int numOccupiedBlocks = 0;
	GLuint Scene;
	GLuint CanvasVAO, CanvasVertVBO, CanvasTexCoordsVBO;
	GLuint InstanceCubeVBO;
	GLuint numOccupiedBlocks_handle = -1;//buffer contains single number.
	//NOTE : we don't really use numOccupiedBlocks for rendering, so it might seem
	//appropriate to keep this in SDF_Hashtable, but we keep it here for cleanliness.

	GLuint SDF_VolumeBuffer_handle = 0;	//actual sdf voxelblocks
	GLuint sdfvoxels_ssbo_index = 0; //ssbo binding index
	GLuint compactHashTable_handle;	//voxelBlock metadata
	std::unique_ptr<FBO> fbo_front;
	std::unique_ptr<FBO> fbo_back;

	//std::unique_ptr<ShaderProgram> raycast_shader;
	std::unique_ptr<ShaderProgram> tempPassthroughShader;	//Temporary shader for debugging integer texture/ cube-ray intersection etc	
	std::unique_ptr<ShaderProgram> instancedCubeDrawShader;
	//std::unique_ptr<ShaderProgram> drawLinearDepth;	//I forgot what this was!
	const float zNear = 0.1f;
	const float zFar = 5.0f;
	glm::mat4 projMat = glm::perspective(45.0f, 1.3333f, zNear, zFar);
	//---------------DEBUG--------------------------------
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
	//void CreateImageBuffer(); //TODO : not needed. get rid of this
	void printSDFdata();
	void render(const glm::mat4&, const glm::mat4&, const glm::vec3&);

	/**
	 * Draws the Container Blocks twice to determine start/stop ranges
	 * for eventual raycast operation
	 */
	void drawToFrontAndBack(const glm::mat4&);

	/**
	 * Draws Voxel container boxes
	 * TODO : Needs to eventually raycast the SDF
	 */
	void drawSDF(ShaderProgram &, const glm::mat4&);
	//friend void registerGLtoCUDA(SDFRenderer*);
	//SDFRenderer(const SDFRenderer&) = delete;
	//SDFRenderer& operator=(const SDFRenderer&) = delete;
};
#endif // !SDF_RENDERER_H
