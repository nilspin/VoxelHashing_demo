#version 430

layout(location=0)in vec4 positions;
layout(location=1)in vec4 normals;

out INTERFACE_BLOCK_VXOUT {
  vec4 pos;
  vec4 normal_dir;
} vx_out;

void main()
{
  vx_out.normal_dir = normals;
  vx_out.pos = positions;
}
