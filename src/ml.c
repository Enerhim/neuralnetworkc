#include "../include/ml.h"

NeuralNetwork createNetwork(uint64_t noLayers, uint64_t *units,
                            ActivationFunction *activations, uint64_t inputSize,
                            LossFunction loss) {
  NeuralNetwork network;
  network.noLayers = noLayers;
  network.loss = loss;
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
    extendVector(nn->layers[i].bias, &extendedBias);

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
  uint64_t noLayers = network->noLayers;
  uint64_t examples = X_train.rows;
  uint64_t outputs = y_train.cols;
  ActivationFunction last_activation = network->layers[noLayers - 1].activation;

  // Gradient Descent Step
  for (uint64_t k = 0; k < epochs; k++) {
    Matrix *A_cache = malloc(sizeof(Matrix) * noLayers);
    Matrix *Z_cache = malloc(sizeof(Matrix) * noLayers);
    Matrix *delta = malloc(sizeof(Matrix) * noLayers);

    Matrix y_hat = inferenceNN(network, X_train, A_cache, Z_cache, true);

    delta[noLayers - 1] = createMatrix(examples, outputs);

    // Optimizing for softmax and shit
    if (last_activation == softmax && network->loss == crossEntropyLoss) {

      subtractMatrices(y_hat, y_train, &delta[noLayers - 1]);
      scaleMatrix(delta[noLayers - 1], 1.00 / examples, &delta[noLayers - 1]);

    } else if (last_activation != softmax && network->loss == MSELoss) {

      // This is the derivative of cost J wrt A of last layer
      Matrix dA = createMatrix(examples, outputs);
      subtractMatrices(y_hat, y_train, &dA);
      scaleMatrix(dA, 1.00 / examples, &dA);

      Matrix activation_derivative = createMatrix(dA.rows, dA.cols);
      // Implement this for the regular activations
      activationDerivative(last_activation, Z_cache[noLayers - 1],
                           &activation_derivative);

      hadamardProduct(dA, activation_derivative, &delta[noLayers - 1]);

      freeMatrix(&activation_derivative);
      freeMatrix(&dA);

    } else {
      fprintf(stderr, "The loss and last layer activation combination is "
                      "highly non-optimal. ");
      exit(1);
    }

    // Calculating intermediate deltas
    for (int64_t l = noLayers - 2; l >= 0; l--) {
      delta[l] = createMatrix(examples, network->layers[l + 1].weights.cols);
      mulMatrices(delta[l + 1], network->layers[l + 1].weights, &delta[l]);

      Matrix activation_derivative = createMatrix(delta[l].rows, delta[l].cols);
      activationDerivative(network->layers[l].activation, Z_cache[l],
                           &activation_derivative);

      hadamardProduct(delta[l], activation_derivative, &delta[l]);

      freeMatrix(&activation_derivative);
    }

    for (uint64_t l = 0; l < noLayers; l++) {
      Matrix A_prev;
      if (l == 0) {
        A_prev = X_train;
      } else {
        A_prev = A_cache[l - 1];
      }

      Matrix delta_T = createMatrix(delta[l].cols, delta[l].rows);
      transposeMatrix(delta[l], &delta_T);

      Matrix dW = createMatrix(delta_T.rows, A_prev.cols);
      mulMatrices(delta_T, A_prev, &dW);
      // scaleMatrix(dW, 1.00 / examples, &dW);

      Matrix dB = createMatrix(delta[l].cols, 1);
      fillMatrix(&dB, 0.00);
      for (uint64_t p = 0; p < delta[l].rows; p++) {
        Matrix row = createMatrix(1, delta[l].cols);
        getRow(delta[l], p, &row);
        Matrix row_T = createMatrix(row.cols, row.rows);
        transposeMatrix(row, &row_T);
        addMatrices(row_T, dB, &dB);
        freeMatrix(&row);
        freeMatrix(&row_T);
      }
      scaleMatrix(dB, 1.00 / examples, &dB);

      clipGradientByNorm(&dW, 5.0);
      clipGradientByNorm(&dB, 5.0);

      // Gradient Descent
      scaleMatrix(dW, alpha, &dW);
      scaleMatrix(dB, alpha, &dB);

      subtractMatrices(network->layers[l].weights, dW,
                       &network->layers[l].weights);
      subtractMatrices(network->layers[l].bias, dB, &network->layers[l].bias);

      freeMatrix(&dW);
      freeMatrix(&dB);
      freeMatrix(&delta_T);
    }

    float cost = network->loss(y_train, y_hat);
    if (k % 50 == 0)
      printf("Epoch: %" PRIu64 " | Cost: %f\n", k, cost);

    for (uint64_t l = 0; l < noLayers; l++) {
      freeMatrix(&A_cache[l]);
      freeMatrix(&Z_cache[l]);
      freeMatrix(&delta[l]);
    }

    freeMatrix(&y_hat);
    free(A_cache);
    free(Z_cache);
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
