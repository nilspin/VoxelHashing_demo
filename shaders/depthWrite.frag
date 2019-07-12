#version 430

uniform sampler2D prevDepthTexture;
uniform float windowWidth;
uniform float windowHeight;

out vec4 outColor;

void main()	{
	vec2 pos = vec2(gl_FragCoord.x/windowWidth, gl_FragCoord.y/windowHeight);	
	if(gl_FragCoord.z < texture(prevDepthTexture,vec2(pos.x,pos.y)).z) discard; //Manually performing the GL_GREATER depth test for each pixel
}