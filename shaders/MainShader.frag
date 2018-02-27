#version 430

uniform vec3 shadeColor;
out vec4 outColor;

void main()
{

	outColor = vec4(shadeColor,1);

}