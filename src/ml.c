#include "../include/ml.h"

NeuralNetwork createNetwork(uint64_t noLayers, uint64_t *units,
                            ActivationFunction *activations,
                            uint64_t inputSize) {
  NeuralNetwork network;
  network.noLayers = noLayers;
  network.layers = malloc(network.noLayers * sizeof(Layer));
  if (!network.layers) {
    fprintf(stderr, "Error: Failed to allocate %" PRIu64 " layers fo NN",
            noLayers);
    exit(1);
  }

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
  Matrix A = createMatrix(X.cols, X.rows);
  transposeMatrix(X, &A); // Convert to colum major for now

  for (uint64_t i = 0; i < nn->noLayers; i++) {
    uint64_t m = nn->layers[i].weights.rows, n = A.cols;
    Matrix dotProduct = createMatrix(m, n);
    mulMatrices(nn->layers[i].weights, A, &dotProduct);

    Matrix extendedBias = createMatrix(m, n);
    extendVector(nn->layers[i].bias, A.cols, &extendedBias);

    Matrix Z = createMatrix(m, n);
    addMatrices(dotProduct, extendedBias, &Z);

    nn->layers[i].activation(&Z);
    freeMatrix(&A);
    A = Z;

    freeMatrix(&extendedBias);
    freeMatrix(&dotProduct);
  }

  return A;
}

void freeNetwork(NeuralNetwork *network) {
  for (uint64_t i = 0; i < network->noLayers; i++) {
    freeMatrix(&network->layers[i].weights);
    freeMatrix(&network->layers[i].bias);
  }

  free(network->layers);
  network->layers = NULL;
  network->noLayers = 0;
}
