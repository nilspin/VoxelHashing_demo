#version 430

//struct VoxelEntry	{
//	ivec3 pos;
//	int ptr;
//	int offset;
//};

uniform mat4 VP;

in vec3 voxentry;
out vec4 v_position;
flat out ivec3 voxCenter_vert;

void main()	{
	//vec3 pos = vec3(voxentry) * 0.05;
	//gl_Position = VP * vec4(pos.x, pos.y, pos.z, 1.0);
	voxCenter_vert = ivec3(voxentry);
	v_position = VP * vec4(voxentry.x, voxentry.y, voxentry.z, 1.0);
}

//gl_Position = vec4(pos.x, pos.y, pos.z, 1.0);
