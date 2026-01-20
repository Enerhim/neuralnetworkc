#include "../include/math.h"
#include <stdint.h>
#include <stdio.h>

Matrix createMatrix(uint64_t rows, uint64_t cols) {
  Matrix mat;
  mat.cols = cols;
  mat.rows = rows;

  mat.data = (float *)malloc(sizeof(float) * rows * cols);

  return mat;
}

void copyMatrix(Matrix original, Matrix *target) {
  if (original.rows != target->rows || original.cols != target->cols) {
    fprintf(stderr,
            "Error: Unable to copy to target of different size: (%" PRIu64
            " x %" PRIu64 ") and (%" PRIu64 " x %" PRIu64 ")\n",
            original.rows, original.cols, target->rows, target->cols);
    return;
  }

  for (uint64_t i = 0; i < original.cols * original.rows; i++) {
    target->data[i] = original.data[i];
  }
}

void fillMatrix(Matrix *A, float fill_val) {
  for (uint64_t i = 0; i < A->cols * A->rows; i++) {
    A->data[i] = fill_val;
  }
}

void addMatrices(Matrix A, Matrix B, Matrix *result) {
  if (A.rows != B.rows || A.cols != B.cols) {
    fprintf(stderr,
            "Error: Unable to add matrices of different sizes: (%" PRIu64
            " x %" PRIu64 ") and (%" PRIu64 " x %" PRIu64 ")\n",
            A.rows, A.cols, B.rows, B.cols);
    return;
  }

  for (uint64_t i = 0; i < A.cols * A.rows; i++) {
    result->data[i] = A.data[i] + B.data[i];
  }
}

void subtractMatrices(Matrix A, Matrix B, Matrix *result) {
  if (A.rows != B.rows || A.cols != B.cols) {
    fprintf(stderr,
            "Error: Unable to subtract matrices of different sizes: (%" PRIu64
            " x %" PRIu64 ") and (%" PRIu64 " x %" PRIu64 ")\n",
            A.rows, A.cols, B.rows, B.cols);
    return;
  }

  for (uint64_t i = 0; i < A.cols * A.rows; i++) {
    result->data[i] = A.data[i] - B.data[i];
  }
}

void mulMatrices(Matrix A, Matrix B, Matrix *result) {
  if (A.cols != B.rows) {
    fprintf(
        stderr,
        "Error: Unable to multiply matrices of incompatible sizes: (%" PRIu64
        " x %" PRIu64 ") and (%" PRIu64 " x %" PRIu64 ")\n",
        A.rows, A.cols, B.rows, B.cols);
    return;
  }

  if (result->rows != A.rows || result->cols != B.cols) {
    fprintf(stderr,
            "Error: result matrix is of wrong size: (%" PRIu64 " x %" PRIu64
            ")\n",
            result->rows, result->cols);
    return;
  }

  for (uint64_t i = 0; i < A.rows; i++) {
    for (uint64_t j = 0; j < B.cols; j++) {
      for (uint64_t k = 0; k < A.cols; k++) {
        result->data[i * result->cols + j] =
            A.data[i * A.cols + k] * B.data[k * B.cols + j];
      }
    }
  }
}

void freeMatrix(Matrix *A) {
  free(A->data);
  A->data = NULL;
  A->rows = 0;
  A->cols = 0;
}

void printMatrix(Matrix A) {
  printf("[");
  for (uint64_t i = 0; i < A.rows; i++) {
    for (uint64_t j = 0; j < A.cols; j++) {
      printf("%f\t", A.data[i * A.cols + j]);
    }
    if (i != A.rows - 1)
      printf("\n");
  }
  printf("]\n");
}
