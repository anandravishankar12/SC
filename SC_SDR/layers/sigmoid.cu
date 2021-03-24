#pragma once
#include "layer.hpp"

__global__ void ForwardSigmoid(float* Z, int nRowsZ, int nColsZ, float* A)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < nRowsZ * nColsZ)
  {
    A[index] = 1 / (1 + exp(-Z[index]));
  }
}

__global__ void BackwardSigmoid(float* Z, float* dA, int nRowsdZ, int nColsdZ,
    float *dZ)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < nRowsdZ * nColsdZ)
  {
    dZ[index] = 1 / (1 + exp(-Z[index])) * (1 - 1 / (1 + exp(-Z[index]))) *
        dA[index];
  }
}

class Sigmoid : public Layer
{
 public:
  Sigmoid()
  {
    dimBlock = 64;
  }

  Matrix& Forward(Matrix& Z)
  {
    this->Z = Z;
    // Z.CopyDeviceToHost();

    A.AllocateMemory(Z.nRows, Z.nCols);

    int dimGrid;
    if ((Z.nRows * Z.nCols) % dimBlock == 0)
      dimGrid = (Z.nRows * Z.nCols) / dimBlock;
    else
      dimGrid = (Z.nRows * Z.nCols) / dimBlock + 1;

    ForwardSigmoid<<<dimGrid, dimBlock>>>(Z.deviceMat.get(), Z.nRows, Z.nCols,
        A.deviceMat.get());
    CheckErrors(cudaGetLastError(),
        "Sigmoid:: Kernel invocation: ForwardSigmoid");

    // A.CopyDeviceToHost();

    return A;
  }

  Matrix& Backward(Matrix& dA, float lr)
  {
    dZ.AllocateMemory(Z.nRows, Z.nCols);

    int dimGrid;
    if ((dZ.nRows * dZ.nCols) % dimBlock == 0)
      dimGrid = (dZ.nRows * dZ.nCols) / dimBlock;
    else
      dimGrid = (dZ.nRows * dZ.nCols) / dimBlock + 1;

    BackwardSigmoid<<<dimGrid, dimBlock>>>(Z.deviceMat.get(), dA.deviceMat.get(),
        dZ.nRows, dZ.nCols, dZ.deviceMat.get());
    CheckErrors(cudaGetLastError(),
        "Sigmoid:: Kernel invocation: BackwardSigmoid");

    // dZ.CopyDeviceToHost();

    return dZ;
  }

  Matrix& ForwardCPU(Matrix& Z)
  {
    this->Z = Z;

    A.AllocateMemory(Z.nRows, Z.nCols);

    for (int i = 0; i < A.nRows; i++)
    {
      for (int j = 0; j < A.nCols; j++)
      {
        A(i, j) = 1 / (1 + exp(-Z(i, j)));
      }
    }

    A.CopyHostToDevice();
    return A;
  }

  Matrix& BackwardCPU(Matrix& dA, float lr)
  {
    dA.CopyDeviceToHost();
    dZ.AllocateMemory(Z.nRows, Z.nCols);

    for (int i = 0; i < A.nRows; i++)
    {
      for (int j = 0; j < A.nCols; j++)
      {
        dZ(i, j) = 1 / (1 + exp(-Z(i, j))) * (1 - 1 / (1 + exp(-Z(i, j)))) *
          dA(i, j);
      }
    }

    // dZ.CopyHostToDevice();
    return dZ;
  }

 private:
  // Input and its derivative w.r.t. the loss.
  Matrix Z;
  Matrix dZ;

  // Output.
  Matrix A;

  int dimBlock;
};
