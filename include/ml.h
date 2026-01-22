#pragma once
#include "math.h"

typedef struct {
  uint64_t units;
  Matrix weights;
  Matrix bias;
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
void fitNetwork(NeuralNetwork *network, float alpha, Matrix X_train,
                Matrix y_train);
