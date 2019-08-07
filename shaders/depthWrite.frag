#version 430

uniform sampler2D prevDepthTexture;
uniform float windowWidth;
uniform float windowHeight;

flat in int SDFVolumeBasePtr_frag;
layout(location=0) out uvec4 outColor;

void main()	{
	vec2 pos = vec2(gl_FragCoord.x/windowWidth, gl_FragCoord.y/windowHeight);
	if(gl_FragCoord.z < texture(prevDepthTexture,pos).x) discard; //Manually performing the GL_GREATER depth test for each pixel

	outColor = uvec4(SDFVolumeBasePtr_frag, 0, 0, 0);
}
