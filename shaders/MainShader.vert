#version 430

in vec4 positions;
uniform mat4 MVP;
//in vec4 normals;
//out vec3 normal_dir;

void main()
{
  //normal_dir = normals.xyz;
	//mat3 normalMatrix = mat3(transpose(inverse(MVP)));
  //normal_dir = normalize(vec3(p))

  gl_Position = MVP*positions;
  
}
