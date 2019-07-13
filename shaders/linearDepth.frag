#version 430

in vec2 uv;

uniform sampler2D depthTexture;
uniform mat4 VP;

out vec4 outColor;

void main()	{
	vec2 NDCcoord = (vec2(uv)*2) - vec2(1);	
	float depth = texture(depthTexture, uv).x;
	float NDCdepth = depth*2 - 1.0;
	vec4 pos = vec4(NDCcoord.x, NDCcoord.y, NDCdepth, 1.0);
	vec4 worldSpacePos = inverse(VP)*pos;
	worldSpacePos /= worldSpacePos.w;
	outColor = vec4(0, worldSpacePos.z, 0, 1);
}