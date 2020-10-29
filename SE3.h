#ifndef SE3_H
#define SE3_H

#include "EigenUtil.h"

Matrix4x4f SE3Exp(const Vector6f&);
Vector6f SE3Log(const Matrix4x4f&);
Vector6f updateTransform(const Vector6f&, const Vector6f);

#endif
