#version 430

in vec4 positions;
in vec4 normals;
//layout(location=0)in vec4 positions;
//layout(location=1)in vec4 normals;
uniform mat4 MVP;
//uniform mat4 VP;

out vec3 FragPos;
out vec3 Normal;

//out INTERFACE_BLOCK_VXOUT {
//  vec4 pos;
//  vec4 normal_dir;
//} vx_out;

void main()
{
  gl_Position = MVP*positions;
  Normal = normals.xyz;
  FragPos = positions.xyz;
//  vx_out.normal_dir = normals;
//  vx_out.pos = positions;
}
