#include "ffn.cu"
#include "layers/linear.cu"
#include "layers/layer.hpp"
#include "layers/relu.cu"
#include "layers/sigmoid.cu"
#include "layers/cross_entropy.cu"
#include "utils.cu"

int main()
{
  int nBatches = 200;
  int batchSize = 2048;
  Dataset dataset(batchSize, batchSize * nBatches);

  FFN network;
  network.Add(new Linear(2, 20));
  network.Add(new ReLU());
  network.Add(new Linear(20, 1));
  network.Add(new Sigmoid());
  network.Train(dataset /* dataset class */,
                0.1 /* learning rate */,
                10 /* epochs */,
                nBatches /* number of batchs */,
                false /* Whether to use CPU instead of GPU */);
}
