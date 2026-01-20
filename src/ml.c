#include "../include/ml.h"
#include <stdint.h>

NeuralNetwork createNetwork(uint64_t noLayers, uint64_t *units,
                            uint8_t *activations, uint64_t inputSize) {
  NeuralNetwork network;
  network.noLayers = noLayers;
  network.layers = malloc(network.noLayers * sizeof(Layer));

  // Initialize weights and activations for each layer

  // The input layer is not a layer,
  // the 0th layer is the first hidden layer
  // the last layer is the output layer
  for (uint64_t i = 0; i < noLayers; i++) {
    network.layers[i].activation = activations[i];
    network.layers[i].units = units[i];

    uint64_t in, out;
    if (i == 0) {
      in = inputSize;
    } else {
      in = units[i - 1];
    }
    out = units[i];

    network.layers[i].weights = createRandomMatrix(out, in);
  }

  return network;
}

void inferenceNN(NeuralNetwork *nn, Matrix X) {
  transposeMatrix(&X);
  Matrix A = createMatrix(X.rows, X.cols);
  copyMatrix(X, &A);

  for (uint64_t i = 0; i < nn->noLayers; i++) {
  }
}

void freeNetwork(NeuralNetwork *network) {
  for (uint64_t i = 0; i < network->noLayers; i++) {
    freeMatrix(&network->layers[i].weights);
  }

  free(network->layers);
  network->layers = NULL;
  network->noLayers = 0;
}
