#pragma once
#include "math.h"

typedef struct {
  uint64_t units;
  Matrix weights;
  Matrix bias;
  // 0 = Linear, 1 = Sigmoid, 2 = ReLU, 3 = Softmax
  ActivationFunction activation;
} Layer;

typedef struct {
  Layer *layers;
  uint64_t noLayers;
} NeuralNetwork;

NeuralNetwork createNetwork(uint64_t noLayers, uint64_t *units,
                            ActivationFunction *activations,
                            uint64_t inputSize);
Matrix inferenceNN(NeuralNetwork *nn, Matrix input);
void freeNetwork(NeuralNetwork *network);
