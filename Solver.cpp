#include "Solver.h"
#include "termcolor.hpp"
#include <string>
//#include <thrust/fill.h>
//#include <thrust/copy.h>

float alpha = 1.0;
float beta = 0.0f;

extern "C" void CalculateJacobiansAndResiduals(const float4* d_input, const float4* corres, const float4* d_corresNormals,
																							 const float* d_Jac, const float* d_res,
																							 const int pyrLevel, const int width, const int height);
//inline
//float calculate_B(const vec3& n, const vec3& d, const vec3& s)  {
//  glm::vec3 p = vec3(d - s);
//  return glm::dot(p,n);
//}

inline
void PrintMatrixDims(const MatrixXf& M, const std::string& s)
{
  std::cout<<"Size of "<<s<<" matrix is : "<<M.rows()<<"x"<<M.cols()<<"\n";
}

inline
void PrintMatrix(const MatrixXf& M, const std::string& s)
{
  std::cout<<"Matrix "<<s<<" : \n"<<M<<"\n";
}

void Solver::PrintSystem()
{
  std::cout << termcolor::green <<"\nFilled matrix system JTJ | JTr : \n"<< termcolor::reset;
  for(int i=0;i<6;++i)
	{
    for(int j=0;j<6;++j)
		{
      std::cout<<JTJ(i,j) <<" ";
    }
    std::cout<< "| "<<JTr(i)<<"\n";
  }

  cout << termcolor::green<< "Calculated solution vector : \n"<<termcolor::reset;
  cout << estimate <<"\n";
}

//void Solver::CalculateJacobians(MatrixXf& JacMat, const vec3& d, const vec3& n, int index)  {
//  vec3 T = cross(d, n);
//  // Calculate Jacobian for this correspondence. Probably most important piece of code
//  // in entire project
//  JacMat.row(index) << n.x, n.y, n.z, T.x, T.y, T.z ;
//}

void Solver::BuildLinearSystem(const float4* d_input, const float4* d_correspondences, const float4*
		d_correspondenceNormals, const float* d_residuals, const int pyrLevel)
{
  //d_residual = thrust::raw_pointer_cast(&d_residuals[0]);
	size_t height = numBlocks[pyrLevel].y * numThreads[pyrLevel].y;
	size_t width  = numBlocks[pyrLevel].x * numThreads[pyrLevel].x;
  int numCorrPairs = width*height;  //corrImageCoords.size();
  //d_residual.resize(numCorrPairs);
  //thrust::fill(d_Jac.begin(), d_Jac.end(), 0);
  //thrust::fill(d_residual.begin(), d_residual.end(), 0);
  //thrust::fill(d_JTJ.begin(), d_JTJ.end(), 0);
  //thrust::fill(d_JTr.begin(), d_JTr.end(), 0);
  //Jac = MatrixXf(numCorrPairs,6);
  //residual = VectorXf(numCorrPairs);
  //PrintMatrixDims(Jac, std::string("Jac"));
  //PrintMatrixDims(residual, std::string("residual"));
  //PrintMatrixDims(JTJ, std::string("JTJ"));
  //PrintMatrixDims(JTr, std::string("JTr"));

  //auto& J = Jac;  //Jacobian;
  //J.setZero();
  //JTJ.setZero();
  //JTr.setZero();
  //residual.setZero();
  //uint idx = 0;

  //Invoke kernel here
  //std::cout<<"\nCalculating Jacobians and residuals\n";
  CalculateJacobiansAndResiduals(d_input, d_correspondences, d_correspondenceNormals,
																 d_Jac, d_residuals,
																 pyrLevel, width, height);
  checkCudaErrors(cudaDeviceSynchronize());
  //Now Jac and res are populated. Invoke cublas functions to calculate JTJ and JTr
  //------------------JTr-------------
  //std::cout<<"\nCalculating JTr\n";
  //TODO : Set this d_residual_ptr correctly
  cuB_res = cublasSgemv(cuB_handle, CUBLAS_OP_N, 6, numCorrPairs, &alpha, d_Jac/*d_a*/, 6, d_residuals/*d_x*/, 1, &beta,
			d_JTr/*d_y*/, 1);
  //copy back
  cudaMemcpy(JTr.data(), d_JTr, JTr_MAX_SIZE, cudaMemcpyDeviceToHost);
  //std::cout<<JTr<<"\n";

  //------------------JTJ-------------
  //std::cout<<"\nCalculating JTJ\n";
  cuB_res = cublasSsyrk(cuB_handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, 6, numCorrPairs, &alpha, d_Jac/*d_a*/, 6, &beta,
			d_JTJ/*d_c*/, 6); //compute JJT in column-maj order
  //Copy back
  cudaMemcpy(raw_JTJ_matrix.data(), d_JTJ, JTJ_MAX_SIZE, cudaMemcpyDeviceToHost);
  //fill upper matrix
  JTJ = Eigen::Map<Matrix6x6f>(raw_JTJ_matrix.data(), 6,6);
  JTJ = JTJ.selfadjointView<Lower>();
  //std::cout<<JTJ<<"\n";
  checkCudaErrors(cudaDeviceSynchronize());
  //---------------------------------

  //for(auto const& iter : corrImageCoords)  {
  //  float3 s = std::get<0>(iter);
  //  float3 d = std::get<1>(iter);
  //  float3 n = std::get<2>(iter);
  //  float r = std::get<3>(iter);
  //  residual.row(idx) << r;  //std::vector to eigen mat
  //  CalculateJacobians(J, d, n, idx);
  //  idx++;
  //}
  ////We have jacobian and residual. Make a linear system.
  //JTJ = Jac.transpose() * Jac;  //should be 6x6
  //JTr = Jac.transpose() * residual;
  JTJinv = JTJ.inverse();
  update = -(JTJinv * JTr);
  estimate = SE3Log(SE3Exp(update) * SE3Exp(estimate) );

  //SolveJacobianSystem(JTJ, JTr);

  //TotalError = residual.transpose() * residual;

  //Print it
  //PrintMatrix(residual, "residual");
  //PrintMatrix(Jac, "Jac");
  //PrintMatrix(JTJ, "JTJ");
  //PrintMatrix(JTr, "JTr");
  //PrintMatrix(update, "update");
  //Our system is built. Solve it
}

void Solver::SolveJacobianSystem(const Matrix6x6f& JTJ, const Vector6f& JTr)
{
  update.setZero();
  //first check if solution exists
  float det = JTJ.determinant();
  if (std::abs(det) < 1e-6 || std::isnan(det) || std::isinf(det))
	{
      solution_exists = false;
  }
  else
	{
    solution_exists = true;
    // Robust Cholesky decomposition of a matrix with pivoting.
    update = JTJ.ldlt().solve(-JTr);
  }
  estimate = SE3Log(SE3Exp(update) * SE3Exp(estimate) );
}

/*
Matrix4x4f Solver::DelinearizeTransform(const Vector6f& x) {
  Matrix4x4f res; res.setIdentity();

	//Rotation
	Matrix3x3f R = Eigen::AngleAxisf(x[0], Eigen::Vector3f::UnitZ()).toRotationMatrix()*
		Eigen::AngleAxisf(x[1], Eigen::Vector3f::UnitY()).toRotationMatrix()*
		Eigen::AngleAxisf(x[2], Eigen::Vector3f::UnitX()).toRotationMatrix();

  //Translation
	Eigen::Vector3f t = x.segment(3, 3);

	res.block(0, 0, 3, 3) = R;
  res.block(0, 3, 3, 1) = t;

	return res;
}
*/

Solver::Solver()
{
  JAC_MAX_SIZE = 6*numCols*numRows*sizeof(float);
  RES_MAX_SIZE = numCols*numRows*sizeof(float);
  JTJ_MAX_SIZE = 6*6*sizeof(float);
  JTr_MAX_SIZE = 6*sizeof(float);

  checkCudaErrors(cudaMalloc((void**)&d_Jac, JAC_MAX_SIZE));
  checkCudaErrors(cudaMemset(d_Jac, 0, JAC_MAX_SIZE));
  //checkCudaErrors(cudaMalloc((void**)&d_residual, RES_MAX_SIZE));
  //checkCudaErrors(cudaMemset(d_residual, 0, RES_MAX_SIZE));
  checkCudaErrors(cudaMalloc((void**)&d_JTJ, JTJ_MAX_SIZE));
  checkCudaErrors(cudaMemset(d_JTJ, 0, JTJ_MAX_SIZE));
  checkCudaErrors(cudaMalloc((void**)&d_JTr, JTr_MAX_SIZE));
  checkCudaErrors(cudaMemset(d_JTr, 0, JTr_MAX_SIZE));
  //d_Jac.resize(6*numCols*numRows);
  //d_residual.resize(numCols*numRows);
  //d_JTr.resize(6);
  //d_JTJ.resize(6*6);

  //thrust::fill(d_Jac.begin(), d_Jac.end(), (float)0.0f);
  //thrust::fill(d_residual.begin(), d_residual.end(), 0);
  cuB_res = cublasCreate(&cuB_handle);

  //d_Jac_ptr = thrust::raw_pointer_cast(&d_Jac[0]);
  //d_residual_ptr = thrust::raw_pointer_cast(&d_residual[0]);
  //d_JTr_ptr = thrust::raw_pointer_cast(&d_JTr[0]);
  //d_JTJ_ptr = thrust::raw_pointer_cast(&d_JTJ[0]);

  estimate.setZero();
  update.setZero();
  JTJ.setZero();
  JTr.setZero();
  JTJinv.setZero();
  //deltaT.setZero();
}

Solver::~Solver()
{
  checkCudaErrors(cudaFree(d_Jac));
  //checkCudaErrors(cudaFree(d_residual));
  checkCudaErrors(cudaFree(d_JTJ));
  checkCudaErrors(cudaFree(d_JTr));
  cublasDestroy(cuB_handle);
}
