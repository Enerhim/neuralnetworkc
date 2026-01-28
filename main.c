#include "include/math.h"
#include "include/ml.h"
#include "include/mnist.h"
#include "include/pcg.h"

int main(void) {
  pcg32_random_t rng;
  pcg32_srandom_r(&rng, 42u, 54u);

  uint64_t n_train_images, n_test_images, n_train_labels, n_test_labels;
  Matrix X_train = load_mnist_dataset("./fashion_dat/train-images-idx3-ubyte",
                                      &n_train_images, 60000);
  Matrix y_labels = load_mnist_labels("./fashion_dat/train-labels-idx1-ubyte",
                                      &n_train_labels, X_train.rows);
  Matrix X_test = load_mnist_dataset("./fashion_dat/t10k-images-idx3-ubyte",
                                     &n_test_images, 10000);
  Matrix y_test_labels = load_mnist_labels(
      "./fashion_dat/t10k-labels-idx1-ubyte", &n_test_labels, X_test.rows);

  uint64_t units[] = {256, 128, 128, 10};
  ActivationFunction activations[] = {relu, relu, softmax};
  NeuralNetwork net =
      createNetwork(3, units, activations, 784, crossEntropyLoss, &rng);

  loadNetwork(&net, "model_fashion.gay");

  Matrix y_train = mnist_hot_encode(y_labels);
  fitNetwork(&net, 0.1, 500, X_train, y_train, X_test, y_test_labels, 100);

  Matrix y_hat = inferenceNN(&net, X_test, NULL, NULL, false);
  float accuracy = calculate_mnist_accuracy(y_hat, y_test_labels);
  printf("Final Accuracy: %f", accuracy);

  saveNetwork(&net, "model_fashion.gay");

  freeNetwork(&net);
  freeMatrix(&X_train);
  freeMatrix(&y_train);
  freeMatrix(&X_test);
  freeMatrix(&y_test_labels);
  freeMatrix(&y_labels);
  return 0;
}
