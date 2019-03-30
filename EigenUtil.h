#ifndef EIGEN_UTIL
#define EIGEN_UTIL

#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <unsupported/Eigen/MatrixFunctions>

using Matrix6x7f = Eigen::Matrix<float, 6, 7>;
//using Matrix6x6f = Eigen::Matrix<float, 6, 6, Eigen::RowMajor>;
using Matrix6x6f = Eigen::Matrix<float, 6, 6>;
using Matrix4x4f = Eigen::Matrix<float, 4, 4>;
using Matrix3x3f = Eigen::Matrix<float, 3, 3>;
using Vector6f = Eigen::Matrix<float, 6, 1>;
using Vector4f = Eigen::Matrix<float, 4, 1>;


#endif //EIGEN_UTIL
