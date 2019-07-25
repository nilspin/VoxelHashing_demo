#version 430

flat in ivec3 voxCenter;

uniform sampler2D startDepthTex;
uniform sampler2D endDepthTex;
uniform float windowWidth;
uniform float windowHeight;
uniform mat4 invVP;

float zNear = 0.1;
float zFar = 5.0;
const int blockSize = 8;
const float voxelSize = 0.02;
const float INFINITY = 1. / 0.;

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

ivec3 voxel2Block(ivec3 voxel) 	{
	if(voxel.x < 0) voxel.x -= blockSize-1;	//i.e voxelBlockSize -1
	if(voxel.y < 0) voxel.y -= blockSize-1;
	if(voxel.z < 0) voxel.z -= blockSize-1;
	return ivec3(voxel.x/blockSize, voxel.y/blockSize, voxel.z/blockSize);
}

ivec3 world2Voxel(vec3 point)	{
	vec3 p = point/voxelSize;
	ivec3 centerOffset = ivec3(sign(p.x), sign(p.y), sign(p.z));
	ivec3 voxelPos =  ivec3(p + vec3(centerOffset.x*0.5, centerOffset.y*0.5, centerOffset.z*0.5));//return center
	return voxelPos;
}

ivec3 world2Block(const vec3 point)	{
	return voxel2Block(world2Voxel(point));
}

ivec3 block2Voxel(const ivec3 block)	{
	ivec3 voxelPos = ivec3(block.x, block.y, block.z) * blockSize;
	return voxelPos;
}
uint linearizeVoxelPos(const ivec3 pos)	{
	const int blockSize = 8;
	return  pos.z * blockSize * blockSize +
			pos.y * blockSize +
			pos.x;
}


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

vec4 traverse(vec3 start, vec3 stop, ivec3 blockPos)	{
	vec4 col = vec4(0);
	vec3 rayDir = stop - start;
	ivec3 stepSize = ivec3(sign(rayDir.x), sign(rayDir.y), sign(rayDir.z));
	vec3 next_boundary = start + vec3(stepSize);
	next_boundary *= voxelSize;

	//calculate max distance to next barrier
	vec3 tMax = vec3(next_boundary - start)/rayDir;
	vec3 tDelta = vec3(voxelSize/rayDir);
	tDelta *= stepSize;

	ivec3 idStart = ivec3((start.x/voxelSize),(start.y/voxelSize),(start.z/voxelSize));
	ivec3 idStop = ivec3((stop.x/voxelSize),(stop.y/voxelSize),(stop.z/voxelSize));
	ivec3 temp = idStart;

	if (rayDir.x == 0.0f) { tMax.x = INFINITY; tDelta.x = INFINITY; }
	if (next_boundary.x - start.x == 0.0f) { tMax.x = INFINITY; tDelta.x = INFINITY; }

	if (rayDir.y == 0.0f) { tMax.y = INFINITY; tDelta.y = INFINITY; }
	if (next_boundary.y - start.y == 0.0f) { tMax.y = INFINITY; tDelta.y = INFINITY; }

	if (rayDir.z == 0.0f) { tMax.z = INFINITY; tDelta.z = INFINITY; }
	if (next_boundary.z - start.z == 0.0f) { tMax.z = INFINITY; tDelta.z = INFINITY; }

	ivec3 baseVoxel = block2Voxel(blockPos);
	ivec3 currVoxel = baseVoxel + temp;
	uint currVoxIdx = linearizeVoxelPos(currVoxel);

	int iter = 0, maxIter = 20;
	while(temp != idStop)	{

		if(tMax.x < tMax.y && tMax.x < tMax.z)	{
			temp.x += stepSize.x;
			//if(temp.x == idEnd.x) break;
			tMax.x += tDelta.x;
		}
		else if(tMax.z < tMax.y)	{
			temp.z += stepSize.z;
			//if(temp.z == idEnd.z) break;
			tMax.z += tDelta.z;
		}
		else{
			temp.y += stepSize.y;
			//if(temp.y == idEnd.y) break;
			tMax.y += tDelta.y;
		}
	}
	//int idx = blockPos.ptr;
	return col;
}

void main()	{
	//----------------Find worldPos---------------------
	vec2 uv = vec2(gl_FragCoord.x/windowWidth, gl_FragCoord.y/windowHeight);
	//float nearDepth = texture(startDepthTex, uv).x;
	//float farDepth = texture(endDepthTex, uv).x;
	//vec3 wrldPos_start = getWorldSpacePosition(nearDepth, uv);
	//vec3 wrldPos_stop = getWorldSpacePosition(farDepth, uv);
	//float dist = length(wrldPos_stop - wrldPos_start);
	//float d = worldSpacePos.z;
	vec3 temp = getWorldSpacePosition(gl_FragCoord.z, uv);
	//temp /= 0.16;
	//ivec3 voxel_start = world2Voxel(wrldPos_start);
	//ivec3 voxel_stop = world2Voxel(wrldPos_stop);
	//temp---------------
	//ivec3 startBlockPos = world2Block(wrldPos_start);
	//ivec3 stopBlockPos = world2Block(wrldPos_stop);
	// startBlockPos and stopBlockPos should be same for all fragments,
	// not just corner ones
	//if(startBlockPos == stopBlockPos)	{ outColor = vec4(1); }
	//else {
	//	//discard;
	//	outColor = vec4(1,0,0,1);
	//}
	//-------------------
	if(world2Block(temp) == voxCenter )	{ outColor = vec4(1,1,1,1);	}
	//if(gl_FragCoord.xyz == voxCenter )	{ outColor = vec4(1,0,0,1);	}
	else 	{
		discard;
		//outColor = vec4(vec3(dist)*normalize(voxCenter), 1);
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
