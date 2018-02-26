#version 430

uniform sampler2D depthTexture;
in vec2 position;
in ivec2 texCoords;
//out vec2 pos;//texturecoords in 0..1
//out float depth;

uniform mat4 MVP;

void main()
{
	//MVP*vec4(position.x, position.y, 0.0, 1.0);
	float hi = texelFetch(depthTexture, texCoords, 0).x;
	float lo = texelFetch(depthTexture, texCoords, 0).y;
	float depth = hi*512+lo;
	gl_Position = MVP*vec4(position.x, -position.y, -depth, 1.0);
	
	//pos = gl_Position.xyz;	
	//pos = (position.xy + vec2(1,1))/2.0;
}
