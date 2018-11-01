#include <iostream>
#include "LinearSystem.h"
#include <cuda_runtime_api.h>
#include "cuda_helper/helper_cuda.h"
#include "termcolor.hpp"
#include <cstring>

//const double M_PI = 3.14159265358979323846;


extern "C" void buildLinearSystemOnDevice(const float4* d_input, const float4* d_target, const float4* d_targetNormals,
	float* d_generatedMatrixSystem, float* h_generatedMatrixSystem);

void LinearSystem::build(const float4* d_input, const float4* d_correspondence, const float4* d_correspondenceNormal, float mean,
					float meanStdev, int width, int height, Matrix6x6f& ATA, Vector6f& ATb) {

	//First clear memory from previous computation
	checkCudaErrors(cudaMemset(d_generatedMatrixSystem, 0x00, OUTPUT_SIZE * sizeof(float)));
	std::memset(h_accumulated_matrix, 0x00, NUMBLOCKS*SYSTEM_SIZE*sizeof(float));

/*  std::cout<<"matrix system before : \n";
  for(int i = 0; i < NUMBLOCKS; ++i)  {
    std::cout<<i<<") ";
    for(int j=0; j<SYSTEM_SIZE; ++j)  {
      std::cout<<h_accumulated_matrix[SYSTEM_SIZE*i + j]<<"\t";
    }
    std::cout<<")\n";
  }
*/

	buildLinearSystemOnDevice(d_input, d_correspondence, d_correspondenceNormal, d_generatedMatrixSystem, h_accumulated_matrix);

	//Reduce 300 row system to single row
	int k = 0;
	for (int i = 1; i < NUMBLOCKS; ++i) {
		for (int j = 0; j < SYSTEM_SIZE; ++j) {
			h_accumulated_matrix[j] += h_accumulated_matrix[SYSTEM_SIZE*i + j];
		}
	}
  
	std::cout <<termcolor::green << "Generated linear system : \n" <<termcolor::reset;
	for (int i = 0; i < SYSTEM_SIZE; ++i) {
		std::cout << h_accumulated_matrix[i] << " ";
	}
	std::cout << "\n";
  

	//Fill the system matrix
	k = 0;
	for (int i = 0; i < 6; ++i) {
		for (int j = i; j < 6; ++j) {
			ATA(i, j) = h_accumulated_matrix[k++];
		}
	}
	//fill lower triangle
	for (int i = 0; i < 6; ++i) {
		for (int j = i; j < 6; ++j) {
			ATA(j, i) = ATA(i, j);
		}
	}
  //fill ATb
  for(int i=0;i<6;++i)  {
    ATb(i) = h_accumulated_matrix[21 + i];
  }
}

LinearSystem::LinearSystem()
{
  accumulated_matrix.reserve(300);
	checkCudaErrors(cudaMalloc((void**)&d_generatedMatrixSystem, OUTPUT_SIZE * sizeof(float)));
	checkCudaErrors(cudaMemset(d_generatedMatrixSystem, 0x00, OUTPUT_SIZE * sizeof(float)));
	h_accumulated_matrix = new float[NUMBLOCKS*SYSTEM_SIZE];
}

LinearSystem::~LinearSystem()
{
	delete h_accumulated_matrix;
	checkCudaErrors(cudaFree(d_generatedMatrixSystem));
}
