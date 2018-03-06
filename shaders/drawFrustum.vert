#version 430

in vec3 position;
uniform mat4 VP;

void main()
{
    gl_Position = VP*vec4(vec3(position), 1.0);    
}
