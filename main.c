#include "include/math.h"

int main(void) {
  Matrix mat = createMatrix(3, 40);

  fillMatrix(&mat, 1.0);

  softmaxMatrix(&mat);
  printMatrix(mat);

  freeMatrix(&mat);
}
