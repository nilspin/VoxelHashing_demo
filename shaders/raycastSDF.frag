#version 430

layout(std430) buffer Voxels	{
	float sdf;
	uchar weight;
	uchar padding[3];
};

in vec2 uv;
uniform mat4 viewMat;
uniform mat4 projMat;
uniform sampler2D startTex;
uniform sampler2D endTex;

void main()	{
	float startDepth = texture(startTex, uv).x;
	float endDepth = texture(endTex, uv).x;
	vec4 temp = vec4(uv.x, uv.y, startDepth, 1.0);
	vec4 camSpacePos = inverse(projMat) * ray;	
	vec4 worldRayStart = inverse(viewMat) * camSpacePos;
}