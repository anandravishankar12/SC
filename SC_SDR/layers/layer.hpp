#pragma once
#include "../matrix.cu"

class Layer {
 public:
  virtual ~Layer() = 0;

  virtual Matrix& Forward(Matrix& A) = 0;
  virtual Matrix& ForwardCPU(Matrix& A) = 0;

  virtual Matrix& Backward(Matrix& dZ, float lr) = 0;
  virtual Matrix& BackwardCPU(Matrix& dZ, float lr) = 0;
};

inline Layer::~Layer() {}
