#ifndef DEBUGHELPER_HPP
#define DEBUGHELPER_HPP

#define GLM_ENABLE_EXPERIMENTAL

#include <iostream>
#include <array>
#include <fstream>
#include <vector>
#include <tuple>
#include <algorithm>
#include <limits>
#include <cstdint>
#include <stdexcept>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
//#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

using std::vector;
using std::cout;
using std::ofstream;
using std::fill;
using std::array;
using std::vector;
using std::cout;
using glm::vec3;
using glm::vec4;
using glm::ivec2;
using glm::mat4;
using glm::mat3;
using glm::quat;
//using CoordPair = std::tuple<ivec2, ivec2, float>;
//using CoordPair = std::tuple<int, int, float>;

struct CorrPair  {
  float3 src;
  float3 targ;
  float3 targNormal;
  float distance = 0; //between two correspondences
  int dummy = -2; //padding
};

//template<typename T>
//void WriteArrayToFile(const vector<T> h_array, std::string filename) {
//  cout<<"Filename : "<<filename<<"\n";
//  ofstream fout(filename.c_str());
//  for(const T& v : h_array) {
//    fout<<glm::to_string(v)<<"\n";
//  }
//  fout.close();
//}

template<typename T>
void ClearVector(vector<T>& V) {
  fill(V.begin(), V.end(), T(0));
  //for_each(V.begin(), V.end(), [](T& temp){temp=T(0);});
}

void ClearVector(vector<CoordPair>& V) {
  //int minInt = std::numeric_limits<int>::min;
  CoordPair temp = (std::make_tuple((INT_MIN), (INT_MIN), 0));
  //CoordPair temp = (std::make_tuple(ivec2(INT_MIN), ivec2(INT_MIN), 0));
  fill(V.begin(), V.end(), temp);
}

//template<typename T>
//void PrintArray(const vector<T> h_array) {
//  for(const T& v : h_array) {
//    cout<<glm::to_string(v)<<"\n";
//  }
//}

/*
template<typename T>
T *PointerAt(const vector<T> &image, int u, int v) {
  uint index = v*640 + u;
  return (T*)image[index];
}
*/

template<typename T>
void checkEquality(const vector<T>& A, const vector<T>& B)  {
  for(auto i=0; i < A.size();  ++i) {
    if(A[i]!=B[i]){
      std::runtime_error("Mismatch at position "+std::to_string(i));
    }
  }
  cout<<"Arrays are same.\n";
}
#endif
