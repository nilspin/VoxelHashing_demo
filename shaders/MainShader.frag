#version 430

//in float depth;
out vec4 outColor;
//in vec2 pos;
//uniform sampler2D depthTexture;

void main()
{
//	outColor = vec4(0, depth*0.05, 0, 1.0);
//	float detectedDepth = texture2D(depthTexture, pos).r;
//	outColor = vec4(1,0,0,1)*detectedDepth/255;
	outColor = vec4(1,0,0,1);
//	vec2 color = texture2D(depthTexture, pos).rg;
//	float actualDepth = color.g;
//	outColor = vec4(vec3(color.r,color.g,1),1);
//	outColor = texture2D(depthTexture, pos);
}