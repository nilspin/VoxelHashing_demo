#version 430

uniform vec3 frustumColor;
out vec4 outColor;

void main()
{
    outColor = vec4(frustumColor, 1.0);
}
