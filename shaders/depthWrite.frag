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

////sorts input vec by its components
//vec3 sort3(vec3 v)
//{
//	vec3 sorted = v;
//	if(v.x > v.y)
//	{
//		sorted.y = v.x;
//		sorted.x = v.y;
//	}
//	if(sorted.y > sorted.z)
//	{
//		sorted.z = sorted.y;
//		sorted.y = v.z;
//		if(sorted.x > sorted.y)
//		{
//			sorted.y = sorted.x;
//			sorted.x = v.z;
//		}
//	}
//	return sorted;
//}

vec4 getColor(ivec3 voxel, uint basePtr)
{
	uint voxIdx = linearizeVoxelPos(voxel);

	const Voxel vox = Voxels[basePtr + voxIdx];

	float sdf = vox.SDF;
	float weight = vox.WEIGHT;

	float col = abs(truncation - abs(sdf))/truncation;
	//float col = abs(sdf);
	//float col = sdf;
	vec4 color = vec4(0);
	if(sdf < 0.0) {
		color = vec4(col, col, col, weight);
	}
	//else{
	//	color = vec4(0, abs(col), 0, weight);
	//}
	return color;
	//return vec4(vec3(1), 0.01);
}

vec4 calculateColor(vec3 startVec, vec3 endVec, vec3 rayDir, uint voxel_basePtr)
{
	startVec = startVec * 8.0;
	endVec = endVec * 8.0;

	//ivec3 stepSize_temp = ivec3(sign(rayDir.x), sign(rayDir.y), sign(rayDir.z)); ///len;
	//startVec -= vec3(0.5)*stepSize_temp; //add offset
	//endVec += vec3(0.5)*stepSize_temp;

	ivec3 startVox = ivec3(floor(startVec));
	ivec3 endVox = ivec3(floor(endVec));
	//if(startVox == endVox) //discard;
	//{
	//	vec4 t_col = vec4(0);
	//	t_col = getColor(startVox, voxel_basePtr);
	//	return t_col;
	//}

	//new rayDir from discrete voxel locations
	vec3 rayDir_n = vec3(endVox) - vec3(startVox);
	rayDir_n = normalize(rayDir_n);

	ivec3 stepSize = ivec3(sign(rayDir_n.x), sign(rayDir_n.y), sign(rayDir_n.z)); ///len;
	vec3 stepSizef = vec3(stepSize); ///len;

	//startVec += 0.1*rayDir; //startVec = floor(startVec);
	//endVec   -= 0.1*rayDir; //endVec   = floor(endVec);
	//startVec -= vec3(0.5)*stepSize; //add offset
	//endVec += vec3(0.5)*stepSize;

	//voxels

	//ivec3 startVox = ivec3(startVec.x, startVec.y, startVec.z);
	//ivec3 endVox = ivec3(endVec.x, endVec.y, endVec.z);


	//startVox += stepSize; //move starting voxel ahead by one position
	//endVox -= stepSize; //move end voxel behind by one position

	//vec3 next_boundary = vec3(startVec + stepSizef);
	vec3 next_boundary = vec3(startVox) + (1.0f/rayDir_n);
	//vec3 next_boundary = vec3(startVec + (0.5*stepSizef));

	//calculate max distance to next barrier
	vec3 tMax = vec3(next_boundary - vec3(startVox))/rayDir_n;
	vec3 tDelta = vec3((1/rayDir_n.x),(1/rayDir_n.y),(1/rayDir_n.z));
	//tDelta *= stepSize;

	//now traverse
	ivec3 temp = startVox ;//+ ivec3(1, 1, 1);
	int iter = 0, maxIter = 20;
	vec4 col = vec4(0);

	while((temp!= endVox) && (iter < maxIter))
	{
	  //while(iter < maxIter)
		//{
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
	vec3 rayDir = rayEnd - rayStart;
	rayDir = normalize(rayDir); //notice the reversed direction!

	float len = length(rayEnd - rayStart)/ sqrt(2.0);
	vec4 cloud = 	vec4(0, 0, 1, 0.2*len);
	outColor = calculateColor(rayStart, rayEnd, rayDir, VoxelBlockID); //traverses the grid
	//outColor += cloud;
}


