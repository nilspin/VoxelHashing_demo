#ifndef LINEAR_SYSTEM_H
#define LINEAR_SYSTEM_H

#include <vector>
#include <cuda_runtime_api.h>
#include "EigenUtil.h"

const int NUMBLOCKS = 300;
const int SYSTEM_SIZE = 27;
const int OUTPUT_SIZE = SYSTEM_SIZE * NUMBLOCKS;

class System;

class LinearSystem
{
public:
	LinearSystem();
	~LinearSystem();
	void build(const float4* input, const float4* correspondence, const float4* correspondenceNormal, float mean, 
				float meanStdev, int width, int height, Matrix6x6f& ATA, Vector6f& ATb);

private:
	float* d_generatedMatrixSystem;
	float* h_accumulated_matrix;
  std::vector<System> accumulated_matrix;
};

class System {
public:
  float coefficients[SYSTEM_SIZE];
  void reset() {
    for(int i=0;i<SYSTEM_SIZE;++i)  {
      coefficients[i] = 0.0f;
    }
  }
  void print()  {
    for(int i=0;i<SYSTEM_SIZE;++i)  {
      std::cout<<coefficients[i]<<" ";
    }
    std::cout<<"\n";
  }
  void add(const System& sys) {
    for(int i=0;i<SYSTEM_SIZE;++i)  {
      coefficients[i] += sys.coefficients[i];
    }
  }
};
#endif //LINEAR_SYSTEM_H