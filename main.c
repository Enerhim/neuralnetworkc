#include "include/math.h"
#include "include/ml.h"
#include "include/mnist.h"

int main(void) {
  srand(time(NULL));

  uint64_t n_train_images, n_test_images, n_train_labels, n_test_labels;

  Matrix X_train = load_mnist_dataset("./data/train-images-idx3-ubyte",
                                      &n_train_images, 100);
  Matrix y_train = load_mnist_labels("./data/train-labels-idx1-ubyte",
                                     &n_train_labels, X_train.rows);

  uint64_t units[] = {16, 16, 10};
  ActivationFunction activations[] = {relu, relu, softmax};

  NeuralNetwork net = createNetwork(3, units, activations, 784);
  fitNetwork(&net, 0.01, 1000, X_train, y_train);

  printShape(X_train);

  freeNetwork(&net);
  freeMatrix(&X_train);
  freeMatrix(&y_train);
  return 0;
}
