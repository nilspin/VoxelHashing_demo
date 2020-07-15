#version 430

layout(location=0) in vec2 pos;
layout(location=1) in vec2 texcoords;

uniform mat4 invMVP;

out vec2 v_texcoords;
out vec3 near_o;
out vec3 far_o; //object space

void main()	{
	gl_Position = vec4(pos.x, pos.y, 0.0, 1.0);
	
	vec4 near_p = vec4(pos.x, pos.y, -1.0, 1.0);
	vec4 far_p = vec4(pos.x, pos.y, 1.0, 1.0);
	
	vec4 near_o_homo = invMVP * near_p;//object space
	vec4 far_o_homo = invMVP * far_p;
	
	//divide by w before interpolation means linear depth
	// according to https://community.khronos.org/t/ray-origin-through-view-and-projection-matrices/72579/4
	//but I didn't understand
	near_o_homo /= near_o_homo.w;
	far_o_homo /= far_o_homo.w;
	
	//these values will be interpolated 
	near_o = near_o_homo.xyz;
	far_o = far_o_homo.xyz;
	
	
	
	v_texcoords = texcoords;

}
