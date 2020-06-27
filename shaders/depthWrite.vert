#version 430

layout(location=0) in vec2 pos;
layout(location=1) in vec2 texcoords;

//uniform mat4 VP;

out vec2 v_texcoords;

void main()	{
	gl_Position = vec4(pos.x, pos.y, 0.0, 1.0);
	v_texcoords = texcoords;

	//VP * vec4(vec3(position), 1.0);
}

//gl_Position = vec4(pos.x, pos.y, pos.z, 1.0);
