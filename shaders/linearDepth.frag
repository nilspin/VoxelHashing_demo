#version 430

flat in vec3 voxCenter;

uniform sampler2D startDepthTex;
uniform sampler2D endDepthTex;
uniform float windowWidth;
uniform float windowHeight;
uniform mat4 invVP;

float zNear = 0.1;
float zFar = 5.0;

out vec4 outColor;

layout(std430, binding=2) buffer VoxelEntry	{
	ivec3 pos;
	int ptr;
	int offset;
	ivec3 padding;
};

layout(std430, binding=3) buffer Voxels	{
	float sdf;
	uint weight;
};

vec3 getWorldSpacePosition(float depth, vec2 uv)	{

	// Convert screen coordinates to normalized device coordinates (NDC)
    vec4 ndc = vec4(
        (gl_FragCoord.x / windowWidth - 0.5) * 2.0,
        (gl_FragCoord.y / windowHeight - 0.5) * 2.0,
        //(gl_FragCoord.z - 0.5) * 2.0,
        (depth - 0.5) * 2.0,
        1.0);

    // Convert NDC throuch inverse clip coordinates to view coordinates
    vec4 clip = invVP * ndc;
    vec3 vertex = (clip / clip.w).xyz;
	return vertex;

	//vec2 NDCcoord = (vec2(uv)*2) - vec2(1);
	//float NDCdepth = depth*2 - 1.0;
	//vec4 pos = vec4(NDCcoord.x, NDCcoord.y, NDCdepth, 1.0);
	//vec4 worldSpacePos = invVP*pos;
	//worldSpacePos /= worldSpacePos.w;
	//return worldSpacePos.xyz;
}

void main()	{
	//----------------Find worldPos---------------------
	vec2 uv = vec2(gl_FragCoord.x/windowWidth, gl_FragCoord.y/windowHeight);
	float nearDepth = texture(startDepthTex, uv).x;
	float farDepth = texture(endDepthTex, uv).x;
	vec3 wrldPos_start = getWorldSpacePosition(nearDepth, uv);
	vec3 wrldPos_stop = getWorldSpacePosition(farDepth, uv);
	float dist = length(wrldPos_stop - wrldPos_start);
	//float d = worldSpacePos.z;
	vec3 temp = getWorldSpacePosition(gl_FragCoord.z, uv);
	temp /= 0.16;
	if(temp.xy == voxCenter.xy )	{ outColor = vec4(1,0,0,1);	}
	//if(gl_FragCoord.xyz == voxCenter )	{ outColor = vec4(1,0,0,1);	}
	else 	{
		outColor = vec4(0, dist, 0, 1);
	}

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
