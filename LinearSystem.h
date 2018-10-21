#ifndef LINEAR_SYSTEM_H
#define LINEAR_SYSTEM_H

#include <cuda_runtime_api.h>
#include "EigenUtil.h"

const int NUMBLOCKS = 300;
const int OUTPUT_SIZE = 26 * NUMBLOCKS;

class LinearSystem
{
public:
	LinearSystem();
	~LinearSystem();
	void build(const float4* input, const float4* correspondence, const float4* correspondenceNormal, float mean, 
				float meanStdev, Matrix4x4f deltaT, int width, int height, Matrix6x7f& system);

private:
	float* d_generatedMatrixSystem;
	float* h_accumulated_matrix;
};

#endif //LINEAR_SYSTEM_H