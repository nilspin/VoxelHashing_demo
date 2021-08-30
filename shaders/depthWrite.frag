#version 430

const float INFINITY = uintBitsToFloat(0x7F800000);

in vec2 v_texcoords;
out vec4 outColor;

struct Voxel {
	float SDF;
	float WEIGHT;
};
layout(std430, packed, binding=3) readonly buffer SDFVolume	{
	Voxel Voxels[];
};

uniform sampler2D rayHit_start;
uniform sampler2D rayHit_end;
uniform isampler2D VoxelID_tex;

const int blockSize = 8;

//ivec3 voxel2Block(ivec3 voxel) 	{
//	if(voxel.x < 0) voxel.x -= blockSize-1;	//i.e voxelBlockSize -1
//	if(voxel.y < 0) voxel.y -= blockSize-1;
//	if(voxel.z < 0) voxel.z -= blockSize-1;
//	return ivec3(voxel.x/blockSize, voxel.y/blockSize, voxel.z/blockSize);
//}

//int linearizeVoxelPos(const ivec3 pos)	{
//	return  pos.z * blockSize * blockSize +
//			pos.y * blockSize +
//			pos.x;
//}

uint linearizeVoxelPos(const ivec3 pos)
{
	return (pos.z * blockSize * blockSize) +
				 (pos.y * blockSize) +
				 (pos.x);
}

vec4 getColor(ivec3 voxel, uint basePtr)
{
	uint voxIdx = linearizeVoxelPos(voxel);

	const Voxel vox = Voxels[basePtr + voxIdx];

	float sdf = vox.SDF;
	float weight = vox.WEIGHT;

	//if(sdf == 0.0f)	{
	//	//color = vec4(vec3(abs(sdf)), weight);
	//	color = vec4(0);
	//}
	//else {
	//	color = vec4(0,0,1,1);
	//	//color = vec4(vec3(abs(sdf)), weight);
	//	//discard;
	//}
	//vec4 color = vec4(vec3(abs(100*sdf)), weight);
	vec4 color = vec4(0);
	if(sdf > 0.0) {
	color = vec4(vec3(sdf), weight);
	}
	return color;
	//return vec4(vec3(1), 0.01);
}

vec4 calculateColor(ivec3 startVox, ivec3 endVox, vec3 rayDir, uint voxel_basePtr)
{
	//int localStartVox = linearizeVoxelPos(startVox);
	//int localEndVox = linearizeVoxelPos(endVox);

	if(startVox == endVox) discard;

	ivec3 stepSize = ivec3(sign(rayDir.x), sign(rayDir.y), sign(rayDir.z));

	vec3 next_boundary = vec3(startVox + stepSize);

	//calculate max distance to next barrier
	vec3 tMax = vec3(next_boundary - vec3(startVox))/rayDir;
	vec3 tDelta = vec3((1/rayDir.x),(1/rayDir.y),(1/rayDir.z));
	//tDelta *= stepSize;

	//handle case when startVox = endVox
	/*
	if (rayDir.x == 0.0f) { tMax.x = INFINITY; tDelta.x = INFINITY; }
	if (next_boundary.x - startVox.x == 0.0f) { tMax.x = INFINITY; tDelta.x = INFINITY; }

	if (rayDir.y == 0.0f) { tMax.y = INFINITY; tDelta.y = INFINITY; }
	if (next_boundary.y - startVox.y == 0.0f) { tMax.y = INFINITY; tDelta.y = INFINITY; }

	if (rayDir.z == 0.0f) { tMax.z = INFINITY; tDelta.z = INFINITY; }
	if (next_boundary.z - startVox.z == 0.0f) { tMax.z = INFINITY; tDelta.z = INFINITY; }
	*/

	//now traverse
	ivec3 temp = startVox;
	int iter = 0, maxIter = 20;
	vec4 col = vec4(0);

	//while(temp!= endVox)
	//{
	  while(iter <= maxIter)
		{
			if((tMax.x < tMax.y) && (tMax.x < tMax.z))
			{
				temp.x += stepSize.x;
				if(temp.x == endVox.x) break;
				tMax.x += tDelta.x;
			}
			if((tMax.y < tMax.x) && (tMax.y < tMax.z))
			{
				temp.y += stepSize.y;
				if(temp.y == endVox.y) break;
				tMax.y += tDelta.y;
			}
			if((tMax.z < tMax.x) && (tMax.z < tMax.y))
			{
				temp.z += stepSize.z;
				if(temp.z == endVox.z) break;
				tMax.z += tDelta.z;
			}
			col += getColor(temp, voxel_basePtr);
			iter++;
		}
	//}

	return col;
}

void main()
{
	uint VoxelBlockID = texture(VoxelID_tex, v_texcoords).w; //Not actual ID. It is base ptr of 8x8x8 voxel brick
	if(VoxelBlockID == 0) discard;

	vec3 rayStart = texture(rayHit_start, v_texcoords).xyz; //in 0..1
	vec3 rayEnd = texture(rayHit_end, v_texcoords).xyz; //in 0..1
	vec3 rayDir = normalize(rayEnd - rayStart);
	//rayDir *= 8.0;

	ivec3 irayStart = ivec3(rayStart*8); //voxel position on 8x8x8 block
	ivec3 irayEnd = ivec3(rayEnd*8);

	outColor = calculateColor(irayStart, irayEnd, rayDir, VoxelBlockID);
}


