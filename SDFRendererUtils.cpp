#include "SDFRendererUtils.h"
#include "common.h"

using std::unique_ptr;

unique_ptr<FBO> setupFBO_w_intTex()	{
	std::unique_ptr<FBO> fbo = unique_ptr<FBO>(new FBO(windowWidth, windowHeight));
	fbo->initResources();
	return fbo;
}


unique_ptr<ShaderProgram> setupInstancedCubeDrawShader()	{
	unique_ptr<ShaderProgram> instancedCubeDrawShader = unique_ptr<ShaderProgram>(new ShaderProgram());
	instancedCubeDrawShader->initFromFiles("shaders/instancedCube.vert", "shaders/instancedCube.frag");
	instancedCubeDrawShader->addUniform("MVP");
	return instancedCubeDrawShader;
}

unique_ptr<ShaderProgram> setupDepthWriteShader()	{
	unique_ptr<ShaderProgram> depthWriteShader = unique_ptr<ShaderProgram>(new ShaderProgram());
	depthWriteShader->initFromFiles("shaders/depthWrite.vert", "shaders/depthWrite.frag");
	//depthWriteShader->addUniform("VP");
	depthWriteShader->addUniform("VoxelID_tex");
	depthWriteShader->addUniform("rayHit_start");
	depthWriteShader->addUniform("rayHit_end");
	return depthWriteShader;
}

void generateCanvas(GLuint& CanvasVAO, GLuint& CanvasVBO, GLuint& CanvasTexCoordsVBO)	{
	GLfloat canvasVerts[] = {		//DATA
		-1.0f,-1.0f,
		-1.0f, 1.0f,
		1.0f, -1.0f,
		1.0f, 1.0f

	};	//Don't need index data for this peasant mesh!


	GLfloat canvasTexCoords[] = {		//DATA
		0.0f, 0.0f,
		0.0f, 1.0f,
		1.0f, 0.0f,
		1.0f, 1.0f

	};	//Don't need index data for this peasant mesh!

	glGenVertexArrays(1, &CanvasVAO);
		glBindVertexArray(CanvasVAO);
		glGenBuffers(1, &CanvasVBO);
			glEnableVertexAttribArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, CanvasVBO);
			glBufferData(GL_ARRAY_BUFFER, sizeof(canvasVerts), &canvasVerts, GL_STATIC_DRAW);
			glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
		//glVertexAttribPointer(drawLinearDepth->attribute("position"), 2, GL_FLOAT, GL_FALSE, 0, 0);
		//glEnableVertexAttribArray(drawLinearDepth->attribute("position"));
		glGenBuffers(1, &CanvasTexCoordsVBO);
			glEnableVertexAttribArray(1);
			glBindBuffer(GL_ARRAY_BUFFER, CanvasTexCoordsVBO);
			glBufferData(GL_ARRAY_BUFFER, sizeof(canvasTexCoords), &canvasTexCoords, GL_STATIC_DRAW);
			glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void generateUnitCube(GLuint &CubeVAO, GLuint& InstanceCubeVertVBO, GLuint& InstanceCubeColVBO)	{

	float cube_tri_data[] = {
	/* face 1 */
	1.0, 0.0, 0.0,
	0.0, 1.0, 0.0,
	1.0, 1.0, 0.0,

	0.0, 1.0, 0.0,
	1.0, 0.0, 0.0,
	0.0, 0.0, 0.0,

	/* face 2 */
	0.0, 1.0, 1.0,
	1.0, 0.0, 1.0,
	1.0, 1.0, 1.0,

	1.0, 0.0, 1.0,
	0.0, 1.0, 1.0,
	0.0, 0.0, 1.0,

	/* face 3 */
	0.0, 1.0, 0.0,
	0.0, 1.0, 1.0,
	1.0, 1.0, 0.0,

	0.0, 1.0, 1.0,
	1.0, 1.0, 1.0,
	1.0, 1.0, 0.0,

	/* face 4 */
	0.0, 0.0, 1.0,
	0.0, 0.0, 0.0,
	1.0, 0.0, 0.0,

	1.0, 0.0, 1.0,
	0.0, 0.0, 1.0,
	1.0, 0.0, 0.0,

	/* face 5 */
	1.0, 1.0, 0.0,
	1.0, 1.0, 1.0,
	1.0, 0.0, 1.0,

	1.0, 1.0, 0.0,
	1.0, 0.0, 1.0,
	1.0, 0.0, 0.0,

	/* face 6 */
	0.0, 1.0, 1.0,
	0.0, 1.0, 0.0,
	0.0, 0.0, 1.0,

	0.0, 0.0, 1.0,
	0.0, 1.0, 0.0,
	0.0, 0.0, 0.0,
	};

	//glGenVertexArrays(1, &CubeVAO);
	//glBindVertexArray(CubeVAO);
	glGenBuffers(1, &InstanceCubeVertVBO);
	glBindBuffer(GL_ARRAY_BUFFER, InstanceCubeVertVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(cube_tri_data), &cube_tri_data, GL_STATIC_DRAW);
	float cube_col_data[] = {
	/* face 1 */
	0.999, 0.001, 0.001,
	0.001, 0.999, 0.001,
	0.999, 0.999, 0.001,

	0.001, 0.999, 0.001,
	0.999, 0.001, 0.001,
	0.001, 0.001, 0.001,

	/* face 2 */
	0.001, 0.999, 0.999,
	0.999, 0.001, 0.999,
	0.999, 0.999, 0.999,

	0.999, 0.001, 0.999,
	0.001, 0.999, 0.999,
	0.001, 0.001, 0.999,

	/* face 3 */
	0.001, 0.999, 0.001,
	0.001, 0.999, 0.999,
	0.999, 0.999, 0.001,

	0.001, 0.999, 0.999,
	0.999, 0.999, 0.999,
	0.999, 0.999, 0.001,

	/* face 4 */
	0.001, 0.001, 0.999,
	0.001, 0.001, 0.001,
	0.999, 0.001, 0.001,

	0.999, 0.001, 0.999,
	0.001, 0.001, 0.999,
	0.999, 0.001, 0.001,

	/* face 5 */
	0.999, 0.999, 0.001,
	0.999, 0.999, 0.999,
	0.999, 0.001, 0.999,

	0.999, 0.999, 0.001,
	0.999, 0.001, 0.999,
	0.999, 0.001, 0.001,

	/* face 6 */
	0.001, 0.999, 0.999,
	0.001, 0.999, 0.001,
	0.001, 0.001, 0.999,

	0.001, 0.001, 0.999,
	0.001, 0.999, 0.001,
	0.001, 0.001, 0.001,
	};
	glGenBuffers(1, &InstanceCubeColVBO);
	glBindBuffer(GL_ARRAY_BUFFER, InstanceCubeColVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(cube_col_data), &cube_col_data, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//glBindVertexArray(CubeVAO);

}

GLuint setup_Debug_SSBO()	{
	//host
	//std::array<debug_ssbo_struct, windowWidth * windowHeight > tempArray;
	std::vector<debug_ssbo_struct> tempArray;
	tempArray.reserve(windowWidth * windowHeight);
	for(auto &i : tempArray)	{
		i.startPtr = 0;
		i.rayStartPos = make_float3(0.0f, 0.0f, 0.0f);
		i.stopPtr = 0;
		i.rayStopPos = make_float3(0.0f, 0.0f, 0.0f);
	}
	//setup device buffer
	GLuint dbg_ssbo = 0;
	glGenBuffers(1, &dbg_ssbo);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, dbg_ssbo);
	glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(debug_ssbo_struct) * tempArray.size() , &tempArray, GL_DYNAMIC_COPY);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, dbg_ssbo);//binding index = 1
	//TODO : glMapBuffer here and copy data?
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	return dbg_ssbo;
}

