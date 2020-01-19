#version 430

uniform layout(binding=1, rgba8ui) writeonly uimage2D imgTex;
//uniform sampler2D prevDepthTexture;
uniform float windowWidth;
uniform float windowHeight;

//flat in int SDFVolumeBasePtr_frag;
//layout(location=0) out uvec4 outColor;

void main()	{
	ivec2 pos = ivec2(gl_FragCoord.x/windowWidth, gl_FragCoord.y/windowHeight);
	//if(gl_FragCoord.z < texture(prevDepthTexture,pos).x) discard; //Manually performing the GL_GREATER depth test for each pixel

	//Note : no longer need to output to texture!
	//write to image instead
	//outColor = uvec4(SDFVolumeBasePtr_frag, 0, 0, 0);
	imageStore(imgTex, pos, uvec4(255,0,0,255));
}
