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
  LossFunction loss;
} NeuralNetwork;

NeuralNetwork createNetwork(uint64_t noLayers, uint64_t *units,
                            ActivationFunction *activations, uint64_t inputSize,
                            LossFunction loss);
Matrix inferenceNN(NeuralNetwork *nn, Matrix input, Matrix *A_cache,
                   Matrix *B_cache, bool cacheZA);
void freeNetwork(NeuralNetwork *network);
void fitNetwork(NeuralNetwork *network, float alpha, uint64_t epochs,
                Matrix X_train, Matrix y_train);
