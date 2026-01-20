#include "include/ml.h"

int main(void) {
  srand(time(NULL));

  uint64_t units[] = {16, 16, 10};
  uint8_t activations[] = {2, 2, 3};

  NeuralNetwork net = createNetwork(3, units, activations, 784);

  for (uint64_t i = 0; i < net.noLayers; i++) {
    printMatrix(net.layers[i].weights);
  }

  freeNetwork(&net);
}
