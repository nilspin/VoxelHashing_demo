#version 430

in vec2 position;
out vec2 uv;

void main()	{
	uv = (position.xy + vec2(1,1))/2.0;
	gl_Position =	vec4(position.x, position.y, 0, 1); 	
}