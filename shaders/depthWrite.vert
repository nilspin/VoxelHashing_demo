#version 430

//struct VoxelEntry	{
//	ivec3 pos;
//	int ptr;
//	int offset;
//} voxentry;
//in VoxelEntry voxentry;

layout(location=0) in ivec3 position;
layout(location=1) in int PtrId_vert;
//layout(location=2) in int blockOffset;
uniform mat4 VP;

//in vec3 voxentry;
//in int SDFVolumeBasePtr_vert;

out vec4 v_position;

//flat out ivec3 voxCenter_vert;
flat out int PtrID_geom;

void main()	{
	//vec3 pos = vec3(voxentry) * 0.05;
	//gl_Position = VP * vec4(pos.x, pos.y, pos.z, 1.0);
	//SDFVolumeBasePtr_geom = SDFVolumeBasePtr_vert;
	//voxCenter_vert = ivec3(voxentry);
	PtrID_geom = PtrId_vert;
	v_position = VP * vec4(vec3(position), 1.0);
}

//gl_Position = vec4(pos.x, pos.y, pos.z, 1.0);
