#pragma once
#include "layer.hpp"
#include <random>

__global__ void ForwardLinear(float *A, float *W, float *b, int nRowsW,
    int nColsW, int nColsA, float *Z)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float ZValue = 0;

  if (row < nRowsW && col < nColsA)
  {
    for (int i = 0; i < nColsW; i++)
    {
      ZValue += W[row * nColsW + i] * A[i * nColsA + col];
    }
    Z[row * nColsA + col] = ZValue + b[row];
  }
}

__global__ void BackwardLinear(float *dZ, float *W, int nColsW, int nRowsW,
    int nColsdZ, float *dA)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float dAValue = 0;

  if (row < nColsW && col < nColsdZ)
  {
    for (int i = 0; i < nRowsW; i++)
    {
      dAValue += W[i * nColsW + row] * dZ[i * nColsdZ + col];
    }
    dA[row * nColsdZ + col] = dAValue;
  }
}

__global__ void UpdateParamsLinear(float *dZ, float *A, int nRowsdZ,
    int nColsdZ, int nRowsA, float lr, float *W, float *b)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  float dWValue = 0, dbValue = 0;

  if (row < nRowsdZ && col < nRowsA)
  {
    for (int i = 0; i < nColsdZ; i++)
    {
      dWValue += dZ[row * nColsdZ + i] * A[col * nColsdZ + i];
    }
    W[row * nRowsA + col] = W[row * nRowsA + col] - lr * dWValue / nColsdZ;

    if (col == 0)
    {
      for (int i = 0; i < nColsdZ; i++)
      {
        dbValue += dZ[row * nColsdZ + i];
      }
      b[row] = b[row] - lr * dbValue / nColsdZ;
    }
  }
}

class Linear : public Layer
{
 public:
  Linear(size_t inSize, size_t outSize) :
      W(outSize, inSize), b(outSize, 1)
  {
    dimBlockX = 16;
    dimBlockY = 16;
    InitializeParameters();
  }

  ~Linear()
  {
    /* Nothing to do here */
  }

  Matrix& Forward(Matrix& A)
  {
    if (A.nRows != W.nCols)
    {
        std::cerr << "ERROR: Number of rows in the input matrix should be " <<
            "equal to the number of columns of the weight matrix." << std::endl;
    }

    this->A = A;

    // Comment the below line if it's already on the device.
    // this->A.CopyHostToDevice();

    Z.AllocateMemory(W.nRows, A.nCols);

    int dimGridX, dimGridY;

    if (Z.nCols % dimBlockX == 0)
      dimGridX = Z.nCols / dimBlockX;
    else
      dimGridX = Z.nCols / dimBlockX + 1;

    if (Z.nRows % dimBlockY == 0)
      dimGridY = Z.nRows / dimBlockY;
    else
      dimGridY = Z.nRows / dimBlockY + 1;

    dim3 dimBlock(dimBlockX, dimBlockY);
    dim3 dimGrid(dimGridX, dimGridY);

    ForwardLinear<<<dimGrid, dimBlock>>>(A.deviceMat.get(), W.deviceMat.get(),
        b.deviceMat.get(), W.nRows, W.nCols, A.nCols, Z.deviceMat.get());
    CheckErrors(cudaGetLastError(),
        "Linear:: Kernel invocation: ForwardLinear");

    // Comment the below line if it's not needed on the host.
    // Z.CopyDeviceToHost();

    return Z;
  }

  // This function assumes that there has been forward pass before it is called.
  Matrix& Backward(Matrix& dZ, float lr = 0.01)
  {
    dA.AllocateMemory(A.nRows, A.nCols);

    // Comment the below line if it's already on the device.
    // dZ.CopyHostToDevice();

    int dimGridX, dimGridY;

    if (dA.nCols % dimBlockX == 0)
      dimGridX = dA.nCols / dimBlockX;
    else
      dimGridX = dA.nCols / dimBlockX + 1;

    if (dA.nRows % dimBlockY == 0)
      dimGridY = dA.nRows / dimBlockY;
    else
      dimGridY = dA.nRows / dimBlockY + 1;

    dim3 dimBlock(dimBlockX, dimBlockY);
    dim3 dimGrid(dimGridX, dimGridY);

    // W is transposed directly during matrix multiplication.
    BackwardLinear<<<dimGrid, dimBlock>>>(dZ.deviceMat.get(), W.deviceMat.get(),
        W.nCols, W.nRows, dZ.nCols, dA.deviceMat.get());
    CheckErrors(cudaGetLastError(),
        "Linear:: Kernel invocation: BackwardLinear");

    // Comment the below line if it's not needed on the host.
    // dA.CopyDeviceToHost();

    UpdateParameters(dZ, lr);

    return dA;
  }

  void UpdateParameters(Matrix& dZ, float lr)
  {
    int dimGridX, dimGridY;

    if (W.nCols % dimBlockX == 0)
      dimGridX = W.nCols / dimBlockX;
    else
      dimGridX = W.nCols / dimBlockX + 1;

    if (W.nRows % dimBlockY == 0)
      dimGridY = W.nRows / dimBlockY;
    else
      dimGridY = W.nRows / dimBlockY + 1;

    dim3 dimBlock(dimBlockX, dimBlockY);
    dim3 dimGrid(dimGridX, dimGridY);

    // A is transposed directly during matrix multiplication.
    UpdateParamsLinear<<<dimGrid, dimBlock>>>(dZ.deviceMat.get(),
        A.deviceMat.get(), dZ.nRows, dZ.nCols, A.nRows, lr,
        W.deviceMat.get(), b.deviceMat.get());
    CheckErrors(cudaGetLastError(),
        "Linear:: Kernel invocation: UpdateParamsLinear");

    // Check if we constantly need to copy weights to host while training,
    // if not, then comment the below code.
    // W.CopyDeviceToHost();
    // b.CopyDeviceToHost();
  }

  /*
   * CPU implementations of functions for time study.
   */

  Matrix& ForwardCPU(Matrix& A)
  {
    if (A.nRows != W.nCols)
    {
        std::cerr << "ERROR: Number of rows in the input matrix should be " <<
            "equal to the number of columns of the weight matrix." << std::endl;
    }

    // A.CopyDeviceToHost();
    this->A = A;

    Z.AllocateMemory(W.nRows, A.nCols);

    for (int i = 0; i < Z.nRows; i++)
    {
      for (int j = 0; j < Z.nCols; j++)
      {
        Z(i, j) = 0;
        for (int k = 0; k < A.nRows; k++)
        {
          Z(i, j) += W(i, k) * A(k, j);
        }

        Z(i, j) += b[i];
      }
    }

    // Z.CopyHostToDevice();

    return Z;
  }

  Matrix& BackwardCPU(Matrix& dZ, float lr = 0.01)
  {
    // dZ.CopyDeviceToHost();

    dA.AllocateMemory(A.nRows, A.nCols);

    for (int i = 0; i < A.nRows; i++)
    {
      for (int j = 0; j < A.nCols; j++)
      {
        dA(i, j) = 0;
        for (int k = 0; k < W.nRows; k++)
        {
          dA(i, j) += W(k, i) * dZ(k, j);
        }
      }
    }

    UpdateParametersCPU(dZ, lr);

    // dA.CopyHostToDevice();
    return dA;
  }

  void UpdateParametersCPU(Matrix& dZ, float lr)
  {
    float dWValue;
    for (int i = 0; i < W.nRows; i++)
    {
      for (int j = 0; j < W.nCols; j++)
      {
        dWValue = 0;
        for (int k = 0; k < dZ.nCols; k++)
        {
          dWValue += dZ(i, k) * A(j, k);
        }
        W(i, j) -= lr * dWValue / dZ.nCols;
      }
    }

    float dbValue;
    for (int i = 0; i < dZ.nRows; i++)
    {
      dbValue = 0;
      for (int j = 0; j < dZ.nCols; j++)
      {
        dbValue += dZ(i, j);
      }
      b[i] -= lr * dbValue / dZ.nCols;
    }
  }

 private:
  // Parameters.
  Matrix W;
  Matrix b;

  // Input and its derivative w.r.t. the loss.
  Matrix A;
  Matrix dA;

  // Output.
  Matrix Z;

  int dimBlockX, dimBlockY;

  void InitializeParameters()
  {
    std::default_random_engine generator;
    std::normal_distribution<float> normalDist(0.0, 1.0);

    for (int i = 0; i < W.nRows; i++)
    {
      for (int j = 0; j < W.nCols; j++)
      {
        W(i, j) = normalDist(generator) * 0.01;

        // For testing, comment the above and uncomment the below line.
        // W(i, j) = 0.01;
      }
    }

    for (int i = 0; i < W.nRows; i++)
    {
      b[i] = 0;

      // For testing, comment the above and uncomment the below line.
      // b[i] = 0.01;
    }

    W.CopyHostToDevice();
    b.CopyHostToDevice();
  }
};


