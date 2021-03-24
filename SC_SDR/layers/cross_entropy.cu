#pragma once
#include "../matrix.cu"

__global__ void ForwardCrossEntropy(float *output, float *labels,
    int nColsOutput, float *loss)
{
  int col = blockIdx.x;

  float temp = -(labels[col] * logf(output[col]) + logf(1 - output[col])
      * (1 - labels[col]));
  atomicAdd(loss, temp);
}

__global__ void BackwardCrossEntropy(float *output, float *labels,
    int nColsOutput, float *dOutput)
{
  int col = blockIdx.x;

  dOutput[col] = (labels[col] / output[col] - (1 - labels[col]) /
      (1 - output[col])) * -1;
}

class CrossEntropy
{
 public:
  CrossEntropy()
  {
  }

  ~CrossEntropy()
  {
  }

  float Forward(Matrix output, Matrix labels)
  {
    if (output.nCols != labels.nCols)
    {
      std::cerr << "ERROR: Number of columns in the output matrix should " <<
          "be equal to the number of colmns of the labels matrix." << std::endl;
    }

    float* loss;
    CheckErrors(cudaMallocManaged(&loss, sizeof(float)),
        "CrossEntropy::Forward() cudaMalloc : loss");
    *loss = 0.0f;

    ForwardCrossEntropy<<<output.nCols, 1>>>(output.deviceMat.get(),
        labels.deviceMat.get(), output.nCols, loss);

    cudaDeviceSynchronize();
    // https://stackoverflow.com/questions/19193468/why-do-we-need-
    // cudadevicesynchronize-in-kernels-with-device-printf
    CheckErrors(cudaGetLastError(),
        "CrossEntropy:: Kernel invocation: ForwardCrossEntropy");

    lossReturn = *loss;
    CheckErrors(cudaFree(loss), "CrossEntropy::Forward() Cuda free : loss");

    return lossReturn / output.nCols;
  }

  Matrix& Backward(Matrix output, Matrix labels)
  {
    if (output.nCols != labels.nCols)
    {
      std::cerr << "ERROR: Number of columns in the output matrix should " <<
          "be equal to the number of colmns of the labels matrix." << std::endl;
    }

    dOutput.AllocateMemory(output.nRows, output.nCols);

    BackwardCrossEntropy<<<output.nCols, output.nRows>>>(output.deviceMat.get(),
        labels.deviceMat.get(), output.nCols, dOutput.deviceMat.get());
    CheckErrors(cudaGetLastError(),
        "CrossEntropy:: Kernel invocation: BackwardCrossEntropy");

    return dOutput;
  }

 private:
  Matrix dOutput;

  float lossReturn;
};
