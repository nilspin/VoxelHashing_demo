#version 430

layout (location=0) out ivec4 VoxelBlockIDs_tex;
layout (location=1) out vec4 outColor;

flat in uint VoxelBlockIDs_frag;
flat in ivec3 boxPosition_frag;
in vec3 rayHit;

void main()	{
	VoxelBlockIDs_tex = ivec4(boxPosition_frag.xyz, VoxelBlockIDs_frag);
	outColor = vec4(rayHit, 1.0);
}
