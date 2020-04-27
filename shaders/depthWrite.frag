#version 430

layout(std430, binding =1) buffer DbgSSBO	{
	uint startPtr;
	vec3 rayStartPos; //where ray hits the block in world-space
	uint stopPtr;
	vec3 rayStopPos; //where ray exits the block in world-space
} dbg_ssbo;

flat in int PtrID_frag;
//flat in int SDFVolumeBasePtr_frag;
layout(location=0) out vec4 outColor;

void main()	{
	dbg_ssbo.startPtr = PtrID_frag;
	dbg_ssbo.rayStartPos = vec3(gl_FragCoord.x, gl_FragCoord.y, gl_FragCoord.z);
	//if(gl_FragCoord.z < texture(prevDepthTexture,pos).x) discard; //Manually performing the GL_GREATER depth test for each pixel

	//Note : no longer need to output to texture!
	//write to image instead
	outColor = vec4(255, 0, 0, 255);
}
