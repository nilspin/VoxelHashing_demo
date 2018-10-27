#version 430

uniform vec3 ShadeColor;
uniform vec3 LightPos;
uniform vec3 CamPos;

in vec3 FragPos;  
in vec3 Normal;  

out vec4 outColor;

vec3 lightColor = vec3(1,1,1);

void main()
{
  // Ambient
  float ambientStrength = 0.1f;
  vec3 ambient = ambientStrength * lightColor;
  
  // Diffuse 
  vec3 norm = normalize(Normal);
  vec3 lightDir = normalize(LightPos - FragPos);
  float diff = max(dot(norm, lightDir), 0.0);
  vec3 diffuse = diff * lightColor;

  // Specular
  float specularStrength = 0.5f;
  vec3 viewDir = normalize(CamPos - FragPos);
  vec3 reflectDir = reflect(-lightDir, norm);  
  float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
  vec3 specular = specularStrength * spec * lightColor;
  
  vec3 result = (ambient + diffuse + specular) * ShadeColor;

	outColor = vec4(result,1);

}