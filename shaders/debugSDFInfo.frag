#version 430

uniform sampler2D startDepthTex;
uniform sampler2D endDepthTex;
uniform usampler2D SDFVolumeBasePtr_texture;
uniform float windowWidth;
uniform float windowHeight;

//should already be 32byte aligned
layout (std430, binding=1) buffer sdf_debug {
	uint startPtr;
	vec3 rayStartPos;
	uint stopPtr;
	vec3 rayStopPos;
};


void main()	{

