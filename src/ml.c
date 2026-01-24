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

    if (network.layers[i].activation == sigmoid ||
        network.layers[i].activation == softmax ||
        network.layers[i].activation == linear) {
      network.layers[i].weights = createXavierMatrix(out, in, in, out);
    } else {
      network.layers[i].weights = createHeMatrix(out, in, in);
    }
    network.layers[i].bias = createMatrix(out, 1);
    fillMatrix(&network.layers[i].bias, 0.0);
  }

  return network;
}

Matrix inferenceNN(NeuralNetwork *nn, Matrix X, Matrix *A_cache,
                   Matrix *Z_cache, bool cacheZA) {
  Matrix A = createMatrix(X.cols, X.rows);
  transposeMatrix(X, &A); // Convert to colum major for working with weights

  for (uint64_t i = 0; i < nn->noLayers; i++) {
    uint64_t m = nn->layers[i].weights.rows, n = A.cols;
    Matrix dotProduct = createMatrix(m, n);
    mulMatrices(nn->layers[i].weights, A, &dotProduct);

    Matrix extendedBias = createMatrix(m, n);
    extendVector(nn->layers[i].bias, A.cols, &extendedBias);

    Matrix Z = createMatrix(m, n);
    addMatrices(dotProduct, extendedBias, &Z);

    if (cacheZA) {
      Z_cache[i] = createMatrix(X.rows, nn->layers[i].units);
      transposeMatrix(Z, &Z_cache[i]);
    }

    nn->layers[i].activation(&Z);
    freeMatrix(&A);
    A = Z;

    if (cacheZA) {
      A_cache[i] = createMatrix(X.rows, nn->layers[i].units);
      transposeMatrix(A, &A_cache[i]);
    }

    freeMatrix(&extendedBias);
    freeMatrix(&dotProduct);
  }

  Matrix result = createMatrix(A.cols, A.rows);
  transposeMatrix(A, &result);
  freeMatrix(&A);
  return result;
}

void fitNetwork(NeuralNetwork *network, float alpha, Matrix X_train,
                Matrix y_train) {
  uint64_t noLayers = network->noLayers;
  uint64_t m = X_train.rows;
  uint64_t n_input = X_train.cols;

  Matrix *A_cache = malloc(sizeof(Matrix) * noLayers);
  Matrix *Z_cache = malloc(sizeof(Matrix) * noLayers);

  Matrix y_hat = inferenceNN(network, X_train, A_cache, Z_cache, true);

  uint64_t n_output = y_hat.cols;

  // Change to y hat rows if not working
  Matrix oneHotLabels = createMatrix(m, n_output);
  oneHotEncode(y_train, &oneHotLabels);

  Matrix dZ = createMatrix(n_output, m);
  transposeMatrix(oneHotLabels, &dZ);

  Matrix *dW = malloc(sizeof(Matrix) * noLayers);
  Matrix *dB = malloc(sizeof(Matrix) * noLayers);

  for (uint64_t l = noLayers - 1; l > 0; l--) {
    Matrix transpose_A = createMatrix(A_cache[l].cols, A_cache[l].rows);
    transposeMatrix(A_cache[l], &transpose_A);

    subtractMatrices(transpose_A, dZ, &dZ);
    dW[l] = createMatrix(dZ.rows, A_cache[l - 1].cols);

    mulMatrices(dZ, A_cache[l - 1], &dW[l]);
    scaleMatrix(dW[l], (1.00 / X_train.rows), &dW[l]);

    dB[l] = createMatrix(y_hat.cols, 1);

    for (uint64_t m = 0; m < X_train.rows; m++) {
      Matrix dZ_m = createMatrix(dZ.rows, 1);
      for (uint64_t i = 0; i < dZ.rows; i++) {
        dZ_m.data[i] = dZ.data[m * i];
      }
      addMatrices(dB[l], dZ_m, &dB[l]);
      freeMatrix(&dZ_m);
    }

    Matrix dZ_prev = createMatrix(A_cache[l - 1].cols, A_cache[l - 1].rows);
    Matrix W_T = createMatrix(network->layers[l].weights.rows,
                              network->layers[l].weights.cols);

    freeMatrix(&transpose_A);
  }

  freeMatrix(&dZ);
  freeMatrix(&oneHotLabels);
  freeMatrix(&y_hat);
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
