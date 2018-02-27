#version 430

uniform sampler2D depthTexture;
//in vec2 position;
in ivec2 texCoords;
//out vec2 pos;//texturecoords in 0..1
//out float depth;

uniform mat4 MVP;

float fx = 525.0;  // focal length x
float fy = 525.0;  // focal length y
float cx = 319.5;  // optical center x
float cy = 239.5;  // optical center y
float factor = 5000; //for freiburg1 dataset (look here-https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats)

void main()
{
	float X = texCoords.x;
	float Y = texCoords.y;

	float hi = texelFetch(depthTexture, texCoords, 0).x;
	float lo = texelFetch(depthTexture, texCoords, 0).y;
	float depth = hi*512+lo;
	//depth = depth/factor;

	float x = (X - cx) * depth/fx;
	float y = (Y - cy) * depth/fy;

	gl_Position = MVP*vec4(x, -y, -depth, 1.0);
	
	//pos = gl_Position.xyz;	
	//pos = (position.xy + vec2(1,1))/2.0;
}
