#version 430

layout(location=0) in vec3 boxVertices;
layout(location=1) in ivec3 boxPosition;
layout(location=2) in uint VoxelBlockIDs;

uniform mat4 VP;
flat out uint VoxelBlockIDs_frag;

void main()	{
VoxelBlockIDs_frag = VoxelBlockIDs;
	vec3 pos = boxVertices + boxPosition;
	gl_Position = VP * vec4(pos, 1.0) ;
}
