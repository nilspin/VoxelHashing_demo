#ifndef DEBUGHELPER_HPP
#define DEBUGHELPER_HPP

#include <vector>
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>
#include "cuda_helper/helper_cuda.h"
//#include <glm/gtx/string_cast.hpp>

//using glm::vec4;
//using glm::vec3;
using std::cout;
using std::vector;
using std::ofstream;

void WriteDeviceArrayToFile(const float4* d_array, std::string filename, const uint len) {
  cout<<"Filename : "<<filename<<"\n";
  vector<float4> h_array(len);
  //h_array.reserve(len);
  checkCudaErrors(cudaMemcpy(h_array.data(), d_array, len*sizeof(float4), cudaMemcpyDeviceToHost));
  ofstream fout(filename.c_str());
  for(const float4& v : h_array) {
    //fout<<glm::to_string(v.xyz)<<"\n";
    fout<<"vec3("<<v.x<<", "<<v.y<<", "<<v.z<<")\n";
  }
  fout.close();
}

#endif