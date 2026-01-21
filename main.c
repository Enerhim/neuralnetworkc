#include "include/math.h"
#include "include/ml.h"

int main(void) {
  srand(time(NULL));

  uint64_t units[] = {16, 16, 10};
  ActivationFunction activations[] = {reluMatrix, reluMatrix, softmaxMatrix};

  NeuralNetwork net = createNetwork(3, units, activations, 5);

  Matrix test = createRandomMatrix(5, 5);

  Matrix result = inferenceNN(&net, test);

  printMatrix(result);

  freeMatrix(&test);
  freeMatrix(&result);

  freeNetwork(&net);
}
