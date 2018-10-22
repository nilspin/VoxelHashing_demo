#include <iostream>
#include "LinearSystem.h"
#include <cuda_runtime_api.h>
#include "cuda_helper/helper_cuda.h"

const double M_PI = 3.14159265358979323846;


extern "C" void buildLinearSystemOnDevice(const float4* d_input, const float4* d_target, const float4* d_targetNormals,
	float* d_generatedMatrixSystem, float* h_generatedMatrixSystem);

void LinearSystem::build(const float4* d_input, const float4* d_correspondence, const float4* d_correspondenceNormal, float mean,
					float meanStdev, int width, int height, Matrix6x7f& system) {

	buildLinearSystemOnDevice(d_input, d_correspondence, d_correspondenceNormal, d_generatedMatrixSystem, h_accumulated_matrix);

	//Reduce 300 row system to single row
	int k = 0;
	for (int i = 1; i < NUMBLOCKS; ++i) {
		for (int j = 0; j < SYSTEM_SIZE; ++j) {
			h_accumulated_matrix[j] += h_accumulated_matrix[SYSTEM_SIZE*i + j];
		}
	}
	std::cout << "Generated linear system : \n";
	for (int i = 0; i < SYSTEM_SIZE; ++i) {
		std::cout << h_accumulated_matrix[i] << "\t";
	}
	std::cout << "\n";

	//Fill the system matrix
	k = 0;
	for (int i = 0; i < 6; ++i) {
		for (int j = i; j < 6; ++j) {
			system(i, j) = h_accumulated_matrix[k++];
		}
		system(i, 6) = h_accumulated_matrix[21 + i];
	}
	//fill lower triangle
	for (int i = 0; i < 6; ++i) {
		for (int j = i; j < 6; ++j) {
			system(j, i) = system(i, j);
		}
	}
}

LinearSystem::LinearSystem()
{
	checkCudaErrors(cudaMalloc((void**)&d_generatedMatrixSystem, OUTPUT_SIZE * sizeof(float)));
	checkCudaErrors(cudaMemset(d_generatedMatrixSystem, 0x00, OUTPUT_SIZE * sizeof(float)));
	h_accumulated_matrix = new float[NUMBLOCKS*SYSTEM_SIZE];
}

LinearSystem::~LinearSystem()
{
	delete h_accumulated_matrix;
	checkCudaErrors(cudaFree(d_generatedMatrixSystem));
}
