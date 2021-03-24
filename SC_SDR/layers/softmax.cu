#pragma once
#include "layer.hpp"

__global__ void ForwardSoftmax(float *Z, int nColsZ, float *sumExp, float *A)
{
  int row = threadIdx.x;
  int col = blockIdx.x;

  atomicAdd(&sumExp[col], exp(Z[row * nColsZ + col]));

  __syncthreads();

  A[row * nColsZ + col] = exp(Z[row * nColsZ + col]) / sumExp[col];
}

__global__ void BackwardSoftmax(float *A, float *dA, int nColsdZ, float *dZ)
{
  int row = threadIdx.x;
  int col = blockIdx.x;

  dZ[row * nColsdZ + col] = dA[row * nColsdZ + col] * A[row * nColsdZ + col] *
      (1 - A[row * nColsdZ + col]);
}

class Softmax : public Layer
{
 public:
  Softmax()
  {
  }


  Matrix& Forward(Matrix& Z)
  {
    this->Z = Z;
    // Z.CopyDeviceToHost();

    A.AllocateMemory(Z.nRows, Z.nCols);

    Matrix sumExp(1, Z.nCols);

    ForwardSoftmax<<<Z.nCols, Z.nRows>>>(Z.deviceMat.get(), Z.nCols,
        sumExp.deviceMat.get(), A.deviceMat.get());
    CheckErrors(cudaGetLastError(),
        "Softmax:: Kernel invocation: ForwardSoftmax");

    // A.CopyDeviceToHost();

    return A;
  }

  Matrix& Backward(Matrix& dA, float lr)
  {
    dZ.AllocateMemory(Z.nRows, Z.nCols);

    BackwardSoftmax<<<Z.nCols, Z.nRows>>>(A.deviceMat.get(), dA.deviceMat.get(),
        dZ.nCols, dZ.deviceMat.get());
    CheckErrors(cudaGetLastError(),
        "Softmax:: Kernel invocation: BackwardSoftmax");

    // dZ.CopyDeviceToHost();

    return dZ;
  }

 private:
  Matrix Z;
  Matrix dZ;

  Matrix A;
};

