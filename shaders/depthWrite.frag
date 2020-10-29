#version 430

//layout(std430, binding =1) buffer DbgSSBO	{
//	uint startPtr;
//	vec3 rayStartPos; //where ray hits the block in world-space
//	uint stopPtr;
//	vec3 rayStopPos; //where ray exits the block in world-space
//} dbg_ssbo;

//flat in int PtrID_frag;
//flat in int SDFVolumeBasePtr_frag;


in vec2 v_texcoords;
in vec2 positions;

uniform isampler2D VoxelID_tex;
uniform mat4 invMVP;
uniform vec3 camPos;
//uniform mat4 invProjMat;
//uniform mat4 invModelViewMat;

layout(location=0) out vec4 outColor;

vec3 near_o;
vec3 far_o;

void main()	{

	//----------------------------copied from vert shader------------
	vec4 near_p = vec4(positions.x, positions.y, -1.0, 1.0);
	vec4 far_p = vec4(positions.x, positions.y, 1.0, 1.0);
	
	vec4 near_o_homo = invMVP * near_p;//object space
	vec4 far_o_homo = invMVP * far_p;
	
	//divide by w before interpolation means linear depth
	// according to https://community.khronos.org/t/ray-origin-through-view-and-projection-matrices/72579/4
	//but I didn't understand
	near_o_homo /= near_o_homo.w;
	far_o_homo /= far_o_homo.w;
	
	//these values will be interpolated 
	near_o = near_o_homo.xyz;
	far_o = far_o_homo.xyz;
	
	//---------------------------------------------------------------
	//uint ID = texelFetch(VoxelID_tex, ivec2(gl_FragCoord.xy), 0).w;
	uint ID = texture(VoxelID_tex, v_texcoords).w;
	
	ivec3 boxPosition = ivec3(texture(VoxelID_tex, v_texcoords).xyz);
	vec3 size = vec3(1,1,1);
	vec3 boxMax = boxPosition + size;
	vec3 boxMin = boxPosition - size;
	
	//boxMin *= 0.02;
	//boxMax *= 0.02;
	
	//paramaterise the rayS
	vec3 rayStart = near_o - camPos;
	vec3 rayEnd = far_o - camPos;
	vec3 rayDir = normalize(rayEnd - rayStart);

	//From :
	//https://developer.arm.com/documentation/100140/0302/advanced-graphics-techniques/implementing-reflections-with-a-local-cubemap/ray-box-intersection-algorithm
	vec3 t_boxmin = (boxMin - rayStart)/rayDir;
	vec3 t_boxmax = (boxMax - rayStart)/rayDir;
	
	vec3 t_absoluteMax = (rayEnd - rayStart)/rayDir;
	float t_absoluteMax_component = -1;
	t_absoluteMax_component = max(t_absoluteMax.x, t_absoluteMax.y);
	t_absoluteMax_component = max(t_absoluteMax_component, t_absoluteMax.z);
	
	//we require the greater value of the t parameter for the intersection at the min plane
	float t_min = (t_boxmin.x > t_boxmin.y) ? t_boxmin.x : t_boxmin.y;
	t_min = (t_min > t_boxmin.z) ? t_min : t_boxmin.z;
	
	float t_max = (t_boxmax.x < t_boxmax.y) ? t_boxmax.x : t_boxmax.y;
	t_max = (t_max < t_boxmax.z) ? t_max : t_boxmax.z;
	
	//----------------------
	if(ID > 0)	{
		//if(abs(t_min) < abs(t_max))	{
		//if(t_min > 0) {
		if((abs(t_min) < abs(t_max)) && (abs(t_max) < abs(t_absoluteMax_component)))	{
			outColor = vec4((t_absoluteMax_component),0,0,1);
		} else {
			outColor = vec4(0,1,0,1);
		}

	}
		//outColor = vec4(t_min, 0, 0, 1);
	
	
	
}
