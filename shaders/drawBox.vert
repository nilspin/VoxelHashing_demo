#version 430

//struct VoxelEntry	{
//	ivec3 pos;
//	int ptr;
//	int offset;
//};

in ivec3 voxentry;
out vec4 v_position;

void main()	{
	v_position = vec4(voxentry.x, voxentry.y, voxentry.z, 1.0);
}

//gl_Position = vec4(pos.x, pos.y, pos.z, 1.0);
