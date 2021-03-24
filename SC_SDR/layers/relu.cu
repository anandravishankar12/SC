#pragma once
#include "layer.hpp"

__global__ void ForwardReLU(float* Z, int nRowsZ, int nColsZ, float* A)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < nRowsZ * nColsZ)
  {
    if (Z[index] >= 0)
      A[index] = Z[index];
    else
      A[index] = 0;
  }
}

__global__ void BackwardReLU(float* Z, float* dA, int nRowsdZ, int nColsdZ,
    float *dZ)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < nRowsdZ * nColsdZ)
  {
    if (Z[index] >= 0)
      dZ[index] = dA[index];
    else
      dZ[index] = 0;
  }
}

class ReLU : public Layer
{
 public:
  ReLU()
  {
    dimBlock = 64;
  }

  ~ReLU()
  {
    /* Nothing to do here */
  }

  Matrix& Forward(Matrix& Z)
  {
    this->Z = Z;

    A.AllocateMemory(Z.nRows, Z.nCols);

    int dimGrid;
    if ((Z.nRows * Z.nCols) % dimBlock == 0)
      dimGrid = (Z.nRows * Z.nCols) / dimBlock;
    else
      dimGrid = (Z.nRows * Z.nCols) / dimBlock + 1;

    ForwardReLU<<<dimGrid, dimBlock>>>(Z.deviceMat.get(), Z.nRows, Z.nCols,
        A.deviceMat.get());
    CheckErrors(cudaGetLastError(),
        "ReLU:: Kernel invocation: ForwardReLU");

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

    BackwardReLU<<<dimGrid, dimBlock>>>(Z.deviceMat.get(), dA.deviceMat.get(),
        dZ.nRows, dZ.nCols, dZ.deviceMat.get());
    CheckErrors(cudaGetLastError(),
        "ReLU:: Kernel invocation: BackwardReLU");

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
        if (Z(i, j) >= 0)
          A(i, j) = Z(i, j);
        else
          A(i, j) = 0;
      }
    }

    // A.CopyHostToDevice();
    return A;
  }

  Matrix& BackwardCPU(Matrix& dA, float lr)
  {
    dZ.AllocateMemory(Z.nRows, Z.nCols);

    for (int i = 0; i < A.nRows; i++)
    {
      for (int j = 0; j < A.nCols; j++)
      {
        if (Z(i, j) >= 0)
          dZ(i, j) = dA(i, j);
        else
          dZ(i, j) = 0;
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

