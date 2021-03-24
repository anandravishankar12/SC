#pragma once

#include <vector>
#include "layers/layer.hpp"
#include "layers/cross_entropy.cu"
#include "utils.cu"

class FFN
{
 public:
  // Constructor.
  FFN()
  {
  }

  Matrix Forward(Matrix input, bool CPU)
  {
    Matrix layerOutput = input;

    if (CPU)
    {
      for (int i = 0; i < network.size(); i++)
      {
        layerOutput = network[i]->ForwardCPU(layerOutput);
      }
    }
    else
    {
      for (int i = 0; i < network.size(); i++)
      {
        layerOutput = network[i]->Forward(layerOutput);
      }
    }

    output = layerOutput;
    return output;
  }

  void Backward(Matrix output, Matrix labels, float lr, bool CPU)
  {
    dOutput = crossEntropy.Backward(output, labels);

    if (CPU)
    {
      for (int i = network.size() - 1; i >= 0; i--)
      {
        dOutput = network[i]->BackwardCPU(dOutput, lr);
      }
    }
    else
    {
      for (int i = network.size() - 1; i >= 0; i--)
      {
        dOutput = network[i]->Backward(dOutput, lr);
      }
    }
  }

  void Train(Dataset dataset, float lr, int nEpochs, int nBatches, bool CPU)
  {
    for (int epoch = 0; epoch < 10; epoch++)
    {
      float accuracy;
      float loss;
      Matrix output;

      for (int batch = 0; batch < nBatches - 1; batch++)
      {
        output = this->Forward(dataset.DataBatches().at(batch), CPU);
        this->Backward(output, dataset.LabelBatches().at(batch), lr, CPU);
      }

      output = this->Forward(dataset.DataBatches().at(nBatches - 1), CPU);
      output.CopyDeviceToHost();

      loss = crossEntropy.Forward(output,
          dataset.LabelBatches().at(nBatches - 1));
      accuracy = Accuracy(output, dataset.LabelBatches().at(nBatches - 1));

      std::cout << "Epoch: " << epoch << std::endl;
      std::cout << "Test Loss: " << loss << std::endl;
      std::cout << "Test Accuracy: " << accuracy << std::endl;
  }
  }

  void Add(Layer* layer)
  {
    this->network.push_back(layer);
  }


  ~FFN()
  {
    for (auto layer : network)
    {
      delete layer;
    }
  }

 private:
  Matrix output;
  Matrix dOutput;
  CrossEntropy crossEntropy;

  std::vector<Layer*> network;
};
