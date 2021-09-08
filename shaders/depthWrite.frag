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
const float truncation = 2.0;

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

	float col = abs(truncation - abs(sdf))/truncation;
	//float col = sdf;
	vec4 color = vec4(0);
	if(sdf < 0.0) {
		color = vec4(abs(col), 0, 0, weight);
	}else{
		color = vec4(0, abs(col), 0, weight);
	}
	return color;
	//return vec4(vec3(1), 0.01);
}

vec4 calculateColor(vec3 startVec, vec3 endVec, vec3 rayDir, uint voxel_basePtr)
{
	startVec = startVec * 8; //startVec += vec3(0.5);
	endVec = endVec * 8; //endVec += vec3(0.5);
	ivec3 stepSize = ivec3(sign(rayDir.x), sign(rayDir.y), sign(rayDir.z));

	//startVec -= vec3(0.5)*stepSize; //add offset
	//endVec += vec3(0.5)*stepSize;

	//voxels
	ivec3 startVox = ivec3(startVec.x, startVec.y, startVec.z);
	ivec3 endVox = ivec3(endVec.x, endVec.y, endVec.z);
	//if(startVox == endVox) discard;

	//startVox += stepSize; //move starting voxel ahead by one position
	//endVox -= stepSize; //move end voxel behind by one position

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
	ivec3 temp = startVox ;//+ ivec3(1, 1, 1);
	int iter = 0, maxIter = 20;
	vec4 col = vec4(0);

	//while(temp!= endVox)
	//{
	  while(iter < maxIter)
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
			//debug
			//if(iter == maxIter-1) col = vec4(0,0,1,1); //print blue
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
	vec3 rayDir = rayEnd - rayStart; //notice the reversed direction!
	//vec3 rayDir = normalize(rayEnd - rayStart); //notice the reversed direction!

	float len = length(rayStart - rayEnd)/ sqrt(2.0);
	vec4 cloud = 	vec4(0, 0, 1, 0.2*len);
	outColor = calculateColor(rayStart, rayEnd, rayDir, VoxelBlockID); //traverses the grid
	outColor += cloud;
}


