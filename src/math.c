#include "../include/math.h"
#include <math.h>

Matrix createMatrix(uint64_t rows, uint64_t cols) {
  Matrix mat;
  mat.cols = cols;
  mat.rows = rows;

  mat.data = (float *)malloc(sizeof(float) * rows * cols);

  return mat;
}

Matrix createRandomMatrix(uint64_t rows, uint64_t cols) {
  Matrix random_ = createMatrix(rows, cols);
  for (uint64_t i = 0; i < rows * cols; i++) {
    random_.data[i] = (float)rand() / (float)RAND_MAX;
  }
  return random_;
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
  printf("]\n Shape: (%" PRIu64 " x %" PRIu64 "), Total Elements: %" PRIu64
         "\nSize = %zu Bytes",
         A.rows, A.cols, A.rows * A.cols, sizeof(float) * A.rows * A.cols);
}

void transposeMatrix(Matrix *A) {
  Matrix result = createMatrix(A->cols, A->rows);

  for (uint64_t i = 0; i < A->rows; i++) {
    for (uint64_t j = 0; j < A->rows; j++) {
      result.data[j * result.rows + i] = A->data[i * A->cols + j];
    }
  }
  freeMatrix(A);
  *A = createMatrix(result.rows, result.cols);
  copyMatrix(result, A);
}

void reluMatrix(Matrix *A) {
  for (uint64_t i = 0; i < A->rows * A->cols; i++) {
    A->data[i] = MAX(A->data[i], 0.0);
  }
}

void sigmoidMatrix(Matrix *A) {
  for (uint64_t i = 0; i < A->rows * A->cols; i++) {
    A->data[i] = 1.0 / (1 + exp(-A->data[i]));
  }
}

void linearMatrix(Matrix *A) { return; }

void softmaxMatrix(Matrix *A) {
  for (uint64_t i = 0; i < A->rows; i++) {
    float sum = 0.0;

    for (uint64_t j = 0; j < A->cols; j++) {
      sum += exp(A->data[i * A->cols + j]);
    }

    for (uint64_t j = 0; j < A->cols; j++) {
      A->data[i * A->cols + j] = exp(A->data[i * A->cols + j]) / sum;
    }
  }
}
