#include "include/math.h"
#include "include/ml.h"
#include "include/mnist.h"

int main(void) {
  srand(time(NULL));

  uint64_t n_train_images, n_test_images, n_train_labels, n_test_labels;

  Matrix *X_train =
      load_mnist_dataset("./data/train-images-idx3-ubyte", &n_train_images);
  Matrix y_train =
      load_mnist_labels("./data/train-labels-idx1-ubyte", &n_train_labels);

  uint64_t units[] = {16, 16, 10};
  ActivationFunction activations[] = {reluMatrix, reluMatrix, softmaxMatrix};

  NeuralNetwork net = createNetwork(3, units, activations, 5);

  Matrix test = createRandomMatrix(5, 5);

  Matrix result = inferenceNN(&net, test);

  printMatrix(result);

  freeMatrix(&test);
  freeMatrix(&result);
  freeNetwork(&net);

  for (uint64_t i = 0; i < n_train_labels; i++) {
    freeMatrix(&X_train[i]);
  }
  freeMatrix(&y_train);
  free(X_train);

  return 0;
}
