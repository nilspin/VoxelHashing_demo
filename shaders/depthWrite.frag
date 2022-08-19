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
const float truncation = 1.0;

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

	//float col = abs(10.0 - abs(sdf))/truncation;
	float col = abs(sdf);
	//float col = sdf;
	vec4 color = vec4(0);
	if(sdf < 0.0) {
		color = vec4(col, col, col, weight);
	}
	else{
		float c = abs(col);
		color = vec4(0,c,0, 0.01);
	}
	return color;
	//return vec4(vec3(1), 0.01);
}

vec4 calculateColor2(vec3 startVec, vec3 endVec, uint voxel_basePtr)
{
	const float stepSize = 0.5;//1.41421356;

	startVec *= 8.0;
	endVec *= 8.0;

	vec3 ray = endVec - startVec;
	float rayLength = length(ray);
	vec3 stepVector = stepSize * ray / rayLength;

	vec4 col = vec4(0);
	while(rayLength > 0)
	{
		ivec3 startVox = ivec3(floor(startVec));
		col += getColor(startVox, voxel_basePtr);
		startVec += stepVector;
		rayLength -= stepSize;
	}

	return col;

}


void main()
{
	uint VoxelBlockID = texture(VoxelID_tex, v_texcoords).w; //Not actual ID. It is base ptr of 8x8x8 voxel brick
	if(VoxelBlockID == 0) discard;

	vec3 rayStart = texture(rayHit_start, v_texcoords).xyz; //in 0..1
	vec3 rayEnd = texture(rayHit_end, v_texcoords).xyz; //in 0..1
	vec3 rayDir = rayEnd - rayStart;
	//rayDir = normalize(rayDir); //notice the reversed direction!

	float len = length(rayEnd - rayStart)/ sqrt(2.0);
	vec4 cloud = 	vec4(1, 1, 1, 0.2*len);
	//outColor = calculateColor(rayStart, rayEnd, rayDir, VoxelBlockID); //traverses the grid
	outColor = calculateColor2(rayStart, rayEnd, VoxelBlockID); //traverses the grid
	//outColor += cloud;
	
}


