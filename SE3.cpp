#include "SE3.h"

// Takes SE3 group object, linearly approximates rotation, returns projected matrix in real space
Matrix4x4f SE3Exp(const Vector6f& twist) {
  Matrix4x4f M;
  M <<       0, -twist(5), twist(4), twist(0),
       twist(5),       0, -twist(3), twist(1),
      -twist(4), twist(3),       0,  twist(2),
             0,        0,        0,        0;
  return M.exp();
}

// Takes SE3 Lie algebra, projects it to abstract SE3 space
Vector6f SE3Log(const Matrix4x4f& transform)  {
  Matrix4x4f M = transform.log();
  Vector6f twist;
  twist << M(0,3), M(1,3), M(2,3), M(2,1), M(0,2), M(1,0);
  return twist;
}

// Notice. While we store rotation/translations as SE3 group elements, we need to
// project them back to real space in order to compute any meaningful transform between
// them because transform directly in abstract space is not possible
Vector6f updateTransform(const Vector6f& perturbation, const Vector6f prev_estimate)  {
  return SE3Log(SE3Exp(perturbation)*SE3Exp(prev_estimate));
}
