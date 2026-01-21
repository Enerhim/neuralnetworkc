#include "../include/ml.h"

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
    network.layers[i].bias = createRandomMatrix(out, 1);
  }

  return network;
}

Matrix inferenceNN(NeuralNetwork *nn, Matrix X) {
  Matrix A = createMatrix(X.rows, X.cols);
  copyMatrix(X, &A);

  for (uint64_t i = 0; i < nn->noLayers; i++) {
    transposeMatrix(&A);

    Matrix dotProduct = createMatrix(nn->layers[i].weights.rows, A.cols);
    mulMatrices(nn->layers[i].weights, A, &dotProduct);

    Matrix extendedBias = createMatrix(nn->layers[i].bias.rows, A.cols);
    extendVector(nn->layers[i].bias, A.cols, &extendedBias);

    Matrix Z = createMatrix(extendedBias.rows, extendedBias.cols);
    addMatrices(dotProduct, extendedBias, &Z);

    freeMatrix(&A);
    A = createMatrix(Z.rows, Z.cols);

    transposeMatrix(&A);
    switch (nn->layers[i].activation) {
    case 0:
      linearMatrix(&A);
      break;
    case 1:
      sigmoidMatrix(&A);
      break;
    case 2:
      reluMatrix(&A);
      break;
    case 3:
      softmaxMatrix(&A);
      break;
    }

    freeMatrix(&extendedBias);
    freeMatrix(&Z);
    freeMatrix(&dotProduct);
  }

  return A;
}

void freeNetwork(NeuralNetwork *network) {
  for (uint64_t i = 0; i < network->noLayers; i++) {
    freeMatrix(&network->layers[i].weights);
  }

  free(network->layers);
  network->layers = NULL;
  network->noLayers = 0;
}
