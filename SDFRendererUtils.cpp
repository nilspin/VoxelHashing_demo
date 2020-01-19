#include "SDFRendererUtils.h"
#include "common.h"

using std::unique_ptr;

unique_ptr<FBO> setupFBO_w_intTex()	{
	std::unique_ptr<FBO> fbo = unique_ptr<FBO>(new FBO(windowWidth, windowHeight));
	fbo->initIntegerTexture();
	return fbo;
}


unique_ptr<ShaderProgram> setupRaycastShader()	{
	unique_ptr<ShaderProgram> raycast_shader = unique_ptr<ShaderProgram>(new ShaderProgram());
	raycast_shader->initFromFiles("shaders/drawBox.vert", "shaders/drawBox.geom", "shaders/drawBox.frag");
	raycast_shader->addAttribute("voxentry");
	raycast_shader->addUniform("VP");
	return raycast_shader;
}

unique_ptr<ShaderProgram> setupDepthWriteShader()	{
	unique_ptr<ShaderProgram> depthWriteShader = unique_ptr<ShaderProgram>(new ShaderProgram());
	depthWriteShader->initFromFiles("shaders/depthWrite.vert", "shaders/depthWrite.geom", "shaders/depthWrite.frag");
	depthWriteShader->addAttribute("voxentry");
	//TODO : enable following attrib again
	//depthWriteShader->addAttribute("SDFVolumeBasePtr_vert");
	depthWriteShader->addUniform("VP");
	//depthWriteShader->addUniform("imgTex");
	//depthWriteShader->addUniform("prevDepthTexture");
	depthWriteShader->addUniform("windowWidth");
	depthWriteShader->addUniform("windowHeight");
	return depthWriteShader;
}

unique_ptr<ShaderProgram> setupDrawLinearDepthShader()	{
	unique_ptr<ShaderProgram> drawLinearDepth = unique_ptr<ShaderProgram>(new ShaderProgram());
	//drawLinearDepth->initFromFiles("shaders/passthrough.vert", "shaders/linearDepth.frag");
	drawLinearDepth->initFromFiles("shaders/raycastSDF.vert", "shaders/raycastSDF.geom", "shaders/raycastSDF.frag");
	drawLinearDepth->addAttribute("voxentry");
	//drawLinearDepth->addAttribute("SDFVolumeBasePtr_vert");
	drawLinearDepth->addUniform("startDepthTex");
	drawLinearDepth->addUniform("endDepthTex");
	drawLinearDepth->addUniform("windowWidth");
	drawLinearDepth->addUniform("windowHeight");
	//drawLinearDepth->addUniform("zNear");
	//drawLinearDepth->addUniform("zFar");
	drawLinearDepth->addUniform("VP");
	drawLinearDepth->addUniform("invVP");
	return drawLinearDepth;
}

