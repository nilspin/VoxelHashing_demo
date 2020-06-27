#version 430

//layout(std430, binding =1) buffer DbgSSBO	{
//	uint startPtr;
//	vec3 rayStartPos; //where ray hits the block in world-space
//	uint stopPtr;
//	vec3 rayStopPos; //where ray exits the block in world-space
//} dbg_ssbo;

//flat in int PtrID_frag;
//flat in int SDFVolumeBasePtr_frag;


in vec2 v_texcoords;
uniform usampler2D VoxelID_tex;

layout(location=0) out vec4 outColor;

void main()	{
	//dbg_ssbo.startPtr = PtrID_frag;
	//dbg_ssbo.rayStartPos = vec3(gl_FragCoord.x, gl_FragCoord.y, gl_FragCoord.z);
	//if(gl_FragCoord.z < texture(prevDepthTexture,pos).x) discard; //Manually performing the GL_GREATER depth test for each pixel

	//uint ID = texelFetch(VoxelID_tex, ivec2(gl_FragCoord.xy), 0).x;
	uint ID = texture(VoxelID_tex, v_texcoords).x;
	if(ID > 0) {
		outColor = vec4(1.0, 0.0, 0.0, 1.0);
	}
	else {
		outColor = vec4(0.0, 1.0, 0.0, 1.0);
	}
}
