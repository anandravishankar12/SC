#include <iostream>

void CheckErrors(cudaError_t error, char const *message)
{
  if (error != cudaSuccess)
  {
    std::cerr << "ERROR: " << message << " : " << cudaGetErrorString(error)
       << std::endl;
    exit(-1);
  }
}
