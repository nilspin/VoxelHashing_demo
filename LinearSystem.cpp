#include "LinearSystem.h"
#include <cuda_runtime_api.h>
#include "cuda_helper/helper_cuda.h"

extern "C" void buildLinearSystemOnDevice(const float4* d_input, const float4* d_target, const float4* d_targetNormals,
	float* d_generatedMatrixSystem);

void LinearSystem::build(const float4* d_input, const float4* d_correspondence, const float4* d_correspondenceNormal, float mean,
					float meanStdev, Matrix4x4f deltaT, int width, int height, Matrix6x7f& system) {

	buildLinearSystemOnDevice(d_input, d_correspondence, d_correspondenceNormal, d_generatedMatrixSystem);
	auto &temp = deltaT;
}

LinearSystem::LinearSystem()
{
	checkCudaErrors(cudaMalloc((void**)&d_generatedMatrixSystem, OUTPUT_SIZE));
	checkCudaErrors(cudaMemset((void**)&d_generatedMatrixSystem, 0, OUTPUT_SIZE * sizeof(float)));
	h_accumulated_matrix = new float[27];
}

LinearSystem::~LinearSystem()
{
	delete h_accumulated_matrix;
	checkCudaErrors(cudaFree(d_generatedMatrixSystem));
}
