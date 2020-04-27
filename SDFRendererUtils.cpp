#include "SDFRendererUtils.h"
#include "common.h"

using std::unique_ptr;

unique_ptr<FBO> setupFBO_w_intTex()	{
	std::unique_ptr<FBO> fbo = unique_ptr<FBO>(new FBO(windowWidth, windowHeight));
	fbo->initIntegerTexture();
	return fbo;
}


//unique_ptr<ShaderProgram> setupRaycastShader()	{
//	unique_ptr<ShaderProgram> raycast_shader = unique_ptr<ShaderProgram>(new ShaderProgram());
//	raycast_shader->initFromFiles("shaders/drawBox.vert", "shaders/drawBox.geom", "shaders/drawBox.frag");
//	raycast_shader->addAttribute("voxentry");
//	raycast_shader->addUniform("VP");
//	//raycast_shader->addUniform("Proj");
//	//raycast_shader->addUniform("invProj");
//	return raycast_shader;
//}

unique_ptr<ShaderProgram> setupDepthWriteShader()	{
	unique_ptr<ShaderProgram> depthWriteShader = unique_ptr<ShaderProgram>(new ShaderProgram());
	depthWriteShader->initFromFiles("shaders/depthWrite.vert", "shaders/depthWrite.geom", "shaders/depthWrite.frag");
	//depthWriteShader->addAttribute("voxentry");
	//TODO : enable following attrib again
	//depthWriteShader->addAttribute("SDFVolumeBasePtr_vert");
	depthWriteShader->addUniform("VP");
	//depthWriteShader->addUniform("imgTex");
	//depthWriteShader->addUniform("prevDepthTexture");
	//depthWriteShader->addUniform("windowWidth");
	//depthWriteShader->addUniform("windowHeight");
	return depthWriteShader;
}

void generateCanvas(GLuint CanvasVAO, GLuint CanvasVBO)	{
	GLfloat canvas[] = {		//DATA
		-1.0f,-1.0f,
		1.0f, -1.0f,
		-1.0f, 1.0f,

		1.0f, -1.0f,
		-1.0f, 1.0f,
		1.0f, 1.0f,
	};	//Don't need index data for this peasant mesh!

	glGenVertexArrays(1, &CanvasVAO);
	glBindVertexArray(CanvasVAO);
	glGenBuffers(1, &CanvasVBO);
	glBindBuffer(GL_ARRAY_BUFFER, CanvasVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(canvas), &canvas, GL_STATIC_DRAW);
	//glVertexAttribPointer(drawLinearDepth->attribute("position"), 2, GL_FLOAT, GL_FALSE, 0, 0);
	//glEnableVertexAttribArray(drawLinearDepth->attribute("position"));
	glBindVertexArray(0);
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
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	return dbg_ssbo;
}
//unique_ptr<ShaderProgram> setupDrawLinearDepthShader()	{
//	unique_ptr<ShaderProgram> drawLinearDepth = unique_ptr<ShaderProgram>(new ShaderProgram());
//	//drawLinearDepth->initFromFiles("shaders/passthrough.vert", "shaders/linearDepth.frag");
//	drawLinearDepth->initFromFiles("shaders/raycastSDF.vert", "shaders/raycastSDF.geom", "shaders/raycastSDF.frag");
//	drawLinearDepth->addAttribute("voxentry");
//	//drawLinearDepth->addAttribute("SDFVolumeBasePtr_vert");
//	drawLinearDepth->addUniform("startDepthTex");
//	drawLinearDepth->addUniform("endDepthTex");
//	drawLinearDepth->addUniform("windowWidth");
//	drawLinearDepth->addUniform("windowHeight");
//	//drawLinearDepth->addUniform("zNear");
//	//drawLinearDepth->addUniform("zFar");
//	drawLinearDepth->addUniform("VP");
//	drawLinearDepth->addUniform("invVP");
//	return drawLinearDepth;
//}

