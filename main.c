#include "include/math.h"
#include "include/ml.h"
#include "include/mnist.h"

int main(void) {
  srand(time(NULL));

  uint64_t n_train_images, n_test_images, n_train_labels, n_test_labels;
  Matrix X_train = load_mnist_dataset("./data/train-images-idx3-ubyte",
                                      &n_train_images, 60000);
  Matrix y_labels = load_mnist_labels("./data/train-labels-idx1-ubyte",
                                      &n_train_labels, X_train.rows);
  Matrix X_test = load_mnist_dataset("./data/t10k-images-idx3-ubyte",
                                     &n_test_images, 10000);
  Matrix y_test_labels = load_mnist_labels("./data/t10k-labels-idx1-ubyte",
                                           &n_test_labels, X_test.rows);

  uint64_t units[] = {16, 16, 10};
  ActivationFunction activations[] = {relu, relu, softmax};
  NeuralNetwork net =
      createNetwork(3, units, activations, 784, crossEntropyLoss);

  Matrix y_train = mnist_hot_encode(y_labels);

  fitNetwork(&net, 0.0005, 1000, X_train, y_train);

  Matrix y_hat = inferenceNN(&net, X_test, NULL, NULL, false);
  float accuracy = calculate_mnist_accuracy(y_hat, y_test_labels);
  printf("Final Accuracy: %f", accuracy);

  freeNetwork(&net);
  freeMatrix(&X_train);
  freeMatrix(&y_train);
  freeMatrix(&X_test);
  freeMatrix(&y_test_labels);
  freeMatrix(&y_labels);
  return 0;
}
