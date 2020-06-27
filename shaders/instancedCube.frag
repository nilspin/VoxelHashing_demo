#version 430

layout (location=0) out uint VoxelBlockIDs_tex;
layout (location=1) out vec4 outColor;

flat in uint VoxelBlockIDs_frag;

void main()	{
	VoxelBlockIDs_tex = VoxelBlockIDs_frag;
	outColor = vec4(1.0, 0, 0, 1.0);
}
