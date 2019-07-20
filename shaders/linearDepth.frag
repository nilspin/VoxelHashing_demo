#version 430

in vec2 uv;

uniform sampler2D startDepthTex;
uniform sampler2D endDepthTex;
//uniform float zNear;
//uniform float zFar;
uniform mat4 VP;

float zNear = 0.1;
float zFar = 5.0;

out vec4 outColor;

vec3 getWorldSpacePosition(float depth)	{

	vec2 NDCcoord = (vec2(uv)*2) - vec2(1);
	float NDCdepth = depth*2 - 1.0;
	vec4 pos = vec4(NDCcoord.x, NDCcoord.y, NDCdepth, 1.0);
	vec4 worldSpacePos = inverse(VP)*pos;
	worldSpacePos /= worldSpacePos.w;
	return worldSpacePos.xyz;
}

void main()	{
	//----------------Find worldPos---------------------
	float nearDepth = texture(startDepthTex, uv).x;
	float farDepth = texture(endDepthTex, uv).x;
	vec3 wrldPos_start = getWorldSpacePosition(nearDepth);
	vec3 wrldPos_stop = getWorldSpacePosition(farDepth);
	float dist = length(wrldPos_stop - wrldPos_start);
	//float d = worldSpacePos.z;
	//if(d < -1) discard;
	outColor = vec4(vec3(dist), 1);

	//------------------Display linear depth---------------
	//float depth = texture(depthTexture, uv).x;
	//float z_n = 2.0 * depth - 1.0;
	//float z_e = (2.0 * zNear)/(zFar + zNear - z_n * (zFar - zNear));
	//outColor = vec4(z_e, z_e, z_e, 1.0);

	//----------------Display nonlinear depth--------------
	//float d = texture(depthTexture, uv).x;
    //outColor = vec4(d, d, d, 1);
	//}
}
