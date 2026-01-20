#include "include/math.h"

int main(void) {
  Matrix mat = createMatrix(3, 2);
  Matrix matu = createMatrix(3, 2);

  fillMatrix(&mat, 2.0);

  copyMatrix(mat, &matu);

  printMatrix(matu);

  freeMatrix(&mat);
  freeMatrix(&matu);
}
