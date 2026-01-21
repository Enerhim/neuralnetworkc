#include "include/math.h"
#include "include/ml.h"

int main(void) {
  srand(time(NULL));

  uint64_t units[] = {16, 16, 10};
  uint8_t activations[] = {2, 2, 3};

  NeuralNetwork net = createNetwork(3, units, activations, 5);

  Matrix test = createMatrix(5, 1);
  fillMatrix(&test, 4.0);

  Matrix result = inferenceNN(&net, test);

  printMatrix(result);

  freeMatrix(&test);
  freeMatrix(&result);

  freeNetwork(&net);
}
