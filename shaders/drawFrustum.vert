#version 430

in vec3 position;
uniform mat4 VP;

float fx = 525.0;  // focal length x
float fy = 525.0;  // focal length y
float cx = 319.5;  // optical center x
float cy = 239.5;  // optical center y

void main()
{
    float Z = position.z*256;
    float X = position.x*Z/fx;	//((position.x - cx)*position.z)/fx;
    float Y = position.y*Z/fy;	//((position.y - cy)*position.z)/fy;
    gl_Position = VP*vec4(X,Y,Z,1.0);

    //gl_Position = VP*vec4(position.x, -position.y, position.z, 1.0);    
}
