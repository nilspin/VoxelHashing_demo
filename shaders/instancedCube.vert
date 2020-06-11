#version 430

layout(location=0) in vec3 boxVertices;
layout(location=1) in ivec3 boxPosition;

uniform mat4 VP;

void main()	{
	vec3 pos = boxVertices + boxPosition;//[gl_InstanceID];
	gl_Position = VP * vec4(pos, 1.0) ;
}
