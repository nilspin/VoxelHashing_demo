#version 430 core

uniform mat4 MVP;
layout (points) in;
layout (line_strip, max_vertices=2) out;

in INTERFACE_BLOCK_VXOUT {
  vec4 pos;
  vec4 normal_dir;
} vertices[];

void main()
{
  vec4 n = vertices[0].normal_dir;

  gl_Position = MVP*(vertices[0].pos);
  EmitVertex();
  gl_Position = MVP*(vertices[0].pos + 100*n.w*vec4(n.xyz,0));
  EmitVertex();

  EndPrimitive();
}