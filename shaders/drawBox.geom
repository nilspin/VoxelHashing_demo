#version 430

layout (points) in;
layout (triangle_strip, max_vertices=4) out;

in ivec4 v_position[];

uniform mat4 viewMat;
uniform mat4 projMat;

void main()	{
	float width = 0.25;
	vec4 origin = vec4(v_position[0].x, v_position[0].y, v_position[0].z, v_position[0].w);
	vec4 p0 = vec4(origin.x - width, origin.y + width, origin.z + width, 1.0);
	vec4 p1 = vec4(origin.x - width, origin.y - width, origin.z + width, 1.0);
	vec4 p2 = vec4(origin.x + width, origin.y + width, origin.z + width, 1.0);
	vec4 p3 = vec4(origin.x + width, origin.y - width, origin.z + width, 1.0);

	gl_Position = projMat * viewMat * p0;
	EmitVertex();

	gl_Position = projMat * viewMat * p1;
	EmitVertex();

	gl_Position = projMat * viewMat * p2;
	EmitVertex();

	gl_Position = projMat * viewMat * p3;
	EmitVertex();

	EndPrimitive();
}