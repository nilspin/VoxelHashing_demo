#version 430

//Draw cube for each vertex.
//Implementation copied from this brilliant answer here : https://stackoverflow.com/a/55136613

layout (points) in;
layout (triangle_strip, max_vertices=16) out;

in vec4 v_position[];
//flat in ivec3 voxCenter_vert[];
//flat out ivec3 voxCenter_frag;

//flat in int SDFVolumeBasePtr_geom[];
//flat out int SDFVolumeBasePtr_frag;

uniform mat4 VP;
//uniform mat4 projMat;

void main()	{
	float width = 0.2;
	vec4 center = v_position[0];
	//SDFVolumeBasePtr_frag = SDFVolumeBasePtr_geom[0];
	vec4 dx = VP[0];
	vec4 dy = VP[1];
	vec4 dz = VP[2];

	vec4 p1 = center;
    vec4 p2 = center + dx;
    vec4 p3 = center + dy;
    vec4 p4 = p2 + dy;
    vec4 p5 = p1 + dz;
    vec4 p6 = p2 + dz;
    vec4 p7 = p3 + dz;
    vec4 p8 = p4 + dz;

	gl_Position = p7;
    EmitVertex();

    gl_Position = p8;
    EmitVertex();

    gl_Position = p5;
    EmitVertex();

    gl_Position = p6;
    EmitVertex();

    gl_Position = p2;
    EmitVertex();

    gl_Position = p8;
    EmitVertex();

    gl_Position = p4;
    EmitVertex();

    gl_Position = p7;
    EmitVertex();

    gl_Position = p3;
    EmitVertex();

    gl_Position = p5;
    EmitVertex();

    gl_Position = p1;
    EmitVertex();

    gl_Position = p2;
    EmitVertex();

    gl_Position = p3;
    EmitVertex();

    gl_Position = p4;
    EmitVertex();

	EndPrimitive();

	gl_Position = p1;
	EmitVertex();

	gl_Position = p5;
	EmitVertex();

	gl_Position = p2;
	EmitVertex();

	gl_Position = p6;
	EmitVertex();

	EndPrimitive();
}
