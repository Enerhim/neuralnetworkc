#include "include/math.h"
#include "include/ml.h"
#include "include/mnist.h"

int main(void) {
  srand(time(NULL));

  uint64_t n_train_images, n_test_images, n_train_labels, n_test_labels;

  Matrix X_train =
      load_mnist_dataset("./data/train-images-idx3-ubyte", &n_train_images, 5);
  Matrix y_train =
      load_mnist_labels("./data/train-labels-idx1-ubyte", &n_train_labels, 5);

  uint64_t units[] = {16, 16, 10};
  ActivationFunction activations[] = {relu, relu, softmax};

  NeuralNetwork net = createNetwork(3, units, activations, 784);

  Matrix logits = inferenceNN(&net, X_train);
  Matrix result = createMatrix(logits.cols, logits.rows);
  transposeMatrix(logits, &result);

  Matrix highestIndices = createMatrix(5, 1);
  getHighestIndexes(result, &highestIndices);

  printMatrix(highestIndices);

  freeNetwork(&net);

  freeMatrix(&logits);
  freeMatrix(&result);
  freeMatrix(&highestIndices);

  freeMatrix(&X_train);
  freeMatrix(&y_train);
  return 0;
}
