#version 430

layout(location=0) in vec3 boxVertices;
layout(location=1) in ivec3 boxPosition;
layout(location=2) in uint VoxelBlockIDs;

uniform mat4 MVP;
flat out uint VoxelBlockIDs_frag;
flat out ivec3 boxPosition_frag;

void main()	{
VoxelBlockIDs_frag = VoxelBlockIDs;
	vec3 pos = boxVertices + boxPosition;
	gl_Position = MVP * vec4(pos, 1.0) ;
}
