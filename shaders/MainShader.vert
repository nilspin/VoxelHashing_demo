#version 430

in vec4 positions;
uniform mat4 MVP;

//TODO: All this from cuda kernel
float fx = 525.0;  // focal length x
float fy = 525.0;  // focal length y
float cx = 319.5;  // optical center x
float cy = 239.5;  // optical center y
float factor = 5000; //for freiburg1 dataset (look here-https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats)

void main()
{
	//gl_Position = MVP*vec4(x, -y, -depth, 1.0);
	gl_Position = MVP*positions;  //or ve4(positions.xyz,1.0)?
}
