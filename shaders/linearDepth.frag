#version 430

in vec2 uv;

uniform sampler2D depthTexture;
//uniform float zNear;
//uniform float zFar;
//uniform mat4 VP;

float zNear = 0.1;
float zFar = 100.0;

out vec4 outColor;

void main()	{
	//----------------Find worldPos---------------------
	//vec2 NDCcoord = (vec2(uv)*2) - vec2(1);
	//float depth = texture(depthTexture, uv).x;
	//float NDCdepth = depth*2 - 1.0;
	//vec4 pos = vec4(NDCcoord.x, NDCcoord.y, NDCdepth, 1.0);
	//vec4 worldSpacePos = inverse(VP)*pos;
	//worldSpacePos /= worldSpacePos.w;
	//outColor = vec4(0, worldSpacePos.z, 0, 1);

	//------------------Display linear depth---------------
	float depth = texture(depthTexture, uv).x;
	float z_n = 2.0 * depth - 1.0;
	float z_e = (2.0 * zNear)/(zFar + zNear - depth * (zFar - zNear));
	outColor = vec4(z_e, z_e, z_e, 1.0);

	//----------------Display nonlinear depth--------------
	//float depth = texture(depthTexture, uv).z;
	//if(gl_FragCoord.x < 1023)	{
	//	outColor = uv.xyyy;
	//}

	//if(gl_FragCoord.x < 1023)	{ outColor = uv.xyyy;}
	//else
	//{
    //vec3 color = texture(depthTexture, uv).xyz;
	//color = vec4(textureLod(depthTexture, uv , 0));
    //outColor = vec4(color.xxx, 1);
	//}
}
