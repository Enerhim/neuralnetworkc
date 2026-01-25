#include "../include/ml.h"
#include <stdint.h>
#include <sys/types.h>

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

void fitNetwork(NeuralNetwork *network, float alpha, uint64_t epochs,
                Matrix X_train, Matrix y_train) {
  uint64_t features = X_train.cols;
  uint64_t examples = X_train.rows;
  uint64_t noLayers = network->noLayers;
  uint64_t outputs = network->layers[noLayers].units;

  ActivationFunction last_activation = network->layers[noLayers - 1].activation;

  // Gradient Descent Step
  for (uint64_t k = 0; k < epochs; k++) {
    Matrix *A_cache = malloc(sizeof(Matrix) * noLayers);
    Matrix *Z_cache = malloc(sizeof(Matrix) * noLayers);

    Matrix y_hat = inferenceNN(network, X_train, A_cache, Z_cache, true);
    Matrix *delta = malloc(sizeof(Matrix) * noLayers);

    // Y_train must be one hot
    // Calculating delta for last layer softmax + cross-entropy
    delta[noLayers - 1] =
        createMatrix(A_cache[noLayers - 1].rows, A_cache[noLayers - 1].cols);
    subtractMatrices(A_cache[noLayers - 1], y_train, &delta[noLayers - 1]);

    for (uint64_t l = noLayers - 2; l >= 0; l--) {
      Matrix W_T = createMatrix(network->layers[l + 1].weights.cols,
                                network->layers[l + 1].weights.rows);
      transposeMatrix(network->layers[l + 1].weights, &W_T);

      delta[l] = createMatrix(W_T.rows, delta[l + 1].cols);
      mulMatrices(W_T, delta[l + 1], &delta[l]);

      ActivationDerivative(network->layers[l].activation, &Z_cache[l]);
      hadamardProduct(delta[l], Z_cache[l], &delta[l]);

      freeMatrix(&W_T);
    }

    Matrix *dW = malloc(sizeof(Matrix) * noLayers);
    Matrix *dB = malloc(sizeof(Matrix) * noLayers);

    for (uint64_t l = noLayers - 1; l > 0; l--) {
      Matrix A_T = createMatrix(A_cache[l - 1].cols, A_cache[l - 1].rows);
      transposeMatrix(A_cache[l - 1], &A_T);

      dW[l] = createMatrix(delta[l].rows, A_T.cols);
      mulMatrices(delta[l], A_T, &dW[l]);
      scaleMatrix(dW[l], 1.00 / examples, &dW[l]);

      dB[l] = createMatrix(delta[l].rows, 1);
      for (uint64_t p = 0; p < delta[l].cols; p++) {
        Matrix column = getColumn(delta[l], p);
        addMatrices(dB[l], column, &dB[l]);
        freeMatrix(&column);
      }
      scaleMatrix(dB[l], 1.00 / examples, &dB[l]);

      freeMatrix(&A_T);
    }
    // dW[0] is special since it requires the train data
    dW[0] = createMatrix(delta[0].rows, X_train.cols);
    mulMatrices(delta[0], X_train, &dW[0]);
    scaleMatrix(dW[0], 1.00 / examples, &dW[0]);
    dB[0] = createMatrix(delta[0].rows, 1);
    for (uint64_t p = 0; p < delta[0].cols; p++) {
      Matrix column = getColumn(delta[0], p);
      addMatrices(dB[0], column, &dB[0]);
      freeMatrix(&column);
    }
    scaleMatrix(dB[0], 1.00 / examples, &dB[0]);

    // GRADIENT DESCEENT YAY
    for (uint64_t l = 0; l < noLayers - 1; l++) {
      scaleMatrix(dW[l], alpha, &dW[l]);
      subtractMatrices(network->layers[l].weights, dW[l],
                       &network->layers[l].weights);

      scaleMatrix(dB[l], alpha, &dB[l]);
      subtractMatrices(network->layers[l].bias, dB[l],
                       &network->layers[l].bias);

      freeMatrix(&dW[l]);
      freeMatrix(&dB[l]);
      freeMatrix(&delta[l]);
      freeMatrix(&A_cache[l]);
      freeMatrix(&Z_cache[l]);
    }

    float cost = crossEntropyLoss(y_train, y_hat);
    printf("Epoch: %" PRIu64 " | Cost: %f", k, cost);

    free(A_cache);
    free(Z_cache);
    free(dW);
    free(dB);
    free(delta);
  }
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
