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

void calculateGradients(Layer *layer, Matrix *w, Matrix *b) {}

// works for cross-entropy and mse loss
void fitNetwork(NeuralNetwork *network, float alpha, Matrix X_train,
                Matrix y_train) {
  for (uint64_t l = 0; l < network->noLayers; l++) {
    float m = network->layers[l].weights.rows,
          n = network->layers[l].weights.cols;
    Matrix dw_l = createMatrix(m, n);
    Matrix db_l = createMatrix(m, 1);
    calculateGradients(&network->layers[l], &dw_l, &db_l);

    Matrix w_l_ = createMatrix(m, n);
    Matrix b_l_ = createMatrix(m, 1);

    for (uint64_t i = 0; i < m; i++) {

      for (uint64_t j = 0; j < n; j++) {
        w_l_.data[i * w_l_.cols + j] -= alpha * dw_l.data[i * dw_l.cols + j];
      }

      b_l_.data[i] -= alpha * db_l.data[i];

      copyMatrix(w_l_, &network->layers[l].weights);
      copyMatrix(b_l_, &network->layers[l].bias);

      freeMatrix(&w_l_);
      freeMatrix(&b_l_);
    }
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
