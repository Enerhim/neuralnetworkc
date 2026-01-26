#include "../include/math.h"

// Matrix Creation

Matrix createMatrix(uint64_t rows, uint64_t cols) {
  Matrix mat;
  mat.cols = cols;
  mat.rows = rows;
  mat.data = (float *)calloc(rows * cols, sizeof(float));

  if (!mat.data) {
    fprintf(stderr,
            "Error: Failed to allocate matrix of size: (%" PRIu64 " x %" PRIu64
            ")",
            rows, cols);
    exit(1);
  }

  return mat;
}

Matrix createRandomMatrix(uint64_t rows, uint64_t cols) {
  Matrix random_ = createMatrix(rows, cols);
  for (uint64_t i = 0; i < rows * cols; i++) {
    random_.data[i] = (float)rand() / (float)RAND_MAX - 0.5;
  }
  return random_;
}

// for sigmoid and tanh activations
Matrix createXavierMatrix(uint64_t rows, uint64_t cols, uint64_t fanIn,
                          uint64_t fanOut) {
  Matrix random_ = createMatrix(rows, cols);
  float limit = sqrtf(6.0 / (fanIn + fanOut));

  for (uint64_t i = 0; i < rows * cols; i++) {
    random_.data[i] = -limit + ((float)rand() / (float)RAND_MAX) * (2 * limit);
  }
  return random_;
}

// for reluuuuu and linear (maybe)
Matrix createHeMatrix(uint64_t rows, uint64_t cols, uint64_t fanIn) {
  Matrix random_ = createMatrix(rows, cols);
  float limit = sqrtf(6.0 / (fanIn));

  for (uint64_t i = 0; i < rows * cols; i++) {
    random_.data[i] = -limit + ((float)rand() / (float)RAND_MAX) * (2 * limit);
  }
  return random_;
}

void copyMatrix(Matrix original, Matrix *target) {
  if (original.rows != target->rows || original.cols != target->cols) {
    fprintf(stderr,
            "Error: Unable to copy to target of different size: (%" PRIu64
            " x %" PRIu64 ") and (%" PRIu64 " x %" PRIu64 ")\n",
            original.rows, original.cols, target->rows, target->cols);
    exit(1);
  }

  for (uint64_t i = 0; i < original.cols * original.rows; i++) {
    target->data[i] = original.data[i];
  }
}

// Matrix Ops

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
    exit(1);
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
    exit(1);
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
    exit(1);
  }

  if (result->rows != A.rows || result->cols != B.cols) {
    fprintf(stderr,
            "Error: result matrix is of wrong size: (%" PRIu64 " x %" PRIu64
            ")\n",
            result->rows, result->cols);
    exit(1);
  }

  for (uint64_t i = 0; i < A.rows; i++) {
    for (uint64_t j = 0; j < B.cols; j++) {
      float sum = 0.0;
      for (uint64_t k = 0; k < A.cols; k++) {
        sum += A.data[i * A.cols + k] * B.data[k * B.cols + j];
      }
      result->data[i * result->cols + j] = sum;
    }
  }
}

void scaleMatrix(Matrix A, float scalar, Matrix *result) {
  for (uint64_t i = 0; i < A.rows * A.cols; i++) {
    result->data[i] = scalar * A.data[i];
  }
}

void extendVector(Matrix A, uint64_t n, Matrix *result) {
  // only row wise extension for now
  if (A.cols != 1) {
    fprintf(stderr,
            "Error: Cannot extend a matrix to a tensor: (%" PRIu64 " x %" PRIu64
            ")",
            A.rows, A.cols);
    exit(1);
  }

  if (result->rows != A.rows || result->cols != A.cols * n) {
    fprintf(stderr,
            "Error: result matrix of wrong shape: A = (%" PRIu64 " x %" PRIu64
            "), Result = (%" PRIu64 " x %" PRIu64 ") ",
            A.rows, A.cols, result->rows, result->cols);
    exit(1);
  }

  for (uint64_t i = 0; i < A.rows; i++) {
    for (uint64_t j = 0; j < n; j++) {
      result->data[i * n + j] = A.data[i];
    }
  }
}

void hadamardProduct(Matrix A, Matrix B, Matrix *result) {
  if (A.rows != B.rows || A.cols != B.cols) {
    fprintf(stderr,
            "Error: Input matrices not of same shape: A = (%" PRIu64
            " x %" PRIu64 "), B = (%" PRIu64 " x %" PRIu64 ") ",
            A.rows, A.cols, B.rows, B.cols);
    exit(1);
  }

  if (A.rows != result->rows || A.cols != result->cols) {
    fprintf(stderr,
            "Error: Result matrix not of same shape: A = (%" PRIu64
            " x %" PRIu64 "), B = (%" PRIu64 " x %" PRIu64 ") ",
            A.rows, A.cols, result->rows, result->cols);
    exit(1);
  }

  for (uint64_t i = 0; i < A.rows; i++) {
    for (uint64_t j = 0; j < A.cols; j++) {
      result->data[i * result->cols + j] =
          A.data[i * A.cols + j] * B.data[i * A.cols + j];
    }
  }
}

Matrix getRow(Matrix A, uint64_t row_index) {
  if (row_index > (A.rows - 1)) {
    fprintf(stderr,
            "Error: Index for getting row is out of bounds. A = (%" PRIu64
            " x %" PRIu64 ") and index = %" PRIu64 "",
            A.rows, A.cols, row_index);
    exit(1);
  }

  Matrix result = createMatrix(1, A.cols);

  for (uint64_t i = 0; i < A.cols; i++) {
    result.data[i] = A.data[A.cols * row_index + i];
  }

  return result;
}

Matrix getColumn(Matrix A, uint64_t column_index) {
  if (column_index > (A.cols - 1)) {
    fprintf(stderr,
            "Error: Index for getting colm is out of bounds. A = (%" PRIu64
            " x %" PRIu64 ") and index = %" PRIu64 "",
            A.rows, A.cols, column_index);
    exit(1);
  }

  Matrix result = createMatrix(A.rows, 1);

  for (uint64_t i = 0; i < A.rows; i++) {
    result.data[i] = A.data[i * A.cols + column_index];
  }

  return result;
}

void transposeMatrix(Matrix A, Matrix *result) {
  if (A.rows != result->cols || A.cols != result->rows) {
    fprintf(
        stderr,
        "Error: Cannot transpose, result matrix is of wrong size. A = (%" PRIu64
        " x %" PRIu64 "), result* = (%" PRIu64 " x %" PRIu64 ")",
        A.rows, A.cols, result->rows, result->cols);
    exit(1);
  }

  for (uint64_t i = 0; i < A.rows; i++) {
    for (uint64_t j = 0; j < A.cols; j++) {
      result->data[j * result->cols + i] = A.data[i * A.cols + j];
    }
  }
}

void freeMatrix(Matrix *A) {
  free(A->data);
  A->data = NULL;
  A->rows = 0;
  A->cols = 0;
}

// Debug

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

void printShape(Matrix A) {
  printf("Shape of matrix is (%" PRIu64 " x %" PRIu64 ")", A.rows, A.cols);
}

// Activations

void relu(Matrix *A) {
  for (uint64_t i = 0; i < A->rows * A->cols; i++) {
    A->data[i] = MAX(A->data[i], 0.0);
  }
}

void sigmoid(Matrix *A) {
  for (uint64_t i = 0; i < A->rows * A->cols; i++) {
    A->data[i] = 1.0 / (1 + exp(-A->data[i]));
  }
}

void linear(Matrix *A) { return; }

void softmax(Matrix *A) {
  for (uint64_t i = 0; i < A->rows; i++) {
    float max_value = -INFINITY;
    for (uint64_t j = 0; j < A->cols; j++) {
      if (A->data[i * A->cols + j] > max_value) {
        max_value = A->data[i * A->cols + j];
      }
    }

    float sum = 0.0;
    for (uint64_t j = 0; j < A->cols; j++) {
      sum += exp(A->data[i * A->cols + j] - max_value);
    }

    for (uint64_t j = 0; j < A->cols; j++) {
      A->data[i * A->cols + j] =
          exp(A->data[i * A->cols + j] - max_value) / sum;
    }
  }
}

// void logMatrix(Matrix *A) {
//   for (uint64_t i = 0; i < A->rows; i++) {
//     for (uint64_t j = 0; j < A->cols; j++) {
//       A->data[i * A->cols + j] = log(A->data[i * A->cols + j]);
//     }
//   }
// }

void activationDerivative(ActivationFunction activation, Matrix Z,
                          Matrix *result) {
  if (Z.rows != result->rows || Z.cols != result->cols) {
    fprintf(stderr,
            "Error: Cannot calculate activation derivaive, result matrix is of "
            "wrong size. Z = (%" PRIu64 " x %" PRIu64 "), result* = (%" PRIu64
            " x %" PRIu64 ")",
            Z.rows, Z.cols, result->rows, result->cols);
    exit(1);
  }

  if (activation == relu) {
    for (uint64_t i = 0; i < Z.rows; i++) {
      for (uint64_t j = 0; j < Z.cols; j++) {
        result->data[i * result->cols + j] =
            Z.data[i * Z.cols + j] > 0 ? 1.0 : 0.0;
      }
    }
  } else if (activation == linear) {
    for (uint64_t i = 0; i < Z.rows; i++) {
      for (uint64_t j = 0; j < Z.cols; j++) {
        result->data[i * result->cols + j] = 1.0;
      }
    }

  } else if (activation == sigmoid) {
    for (uint64_t i = 0; i < Z.rows; i++) {
      for (uint64_t j = 0; j < Z.cols; j++) {
        result->data[i * result->cols + j] =
            (1.00 / (1 + exp(-Z.data[i * Z.cols + j]))) *
            (1 - ((1.00 / (1 + exp(-Z.data[i * Z.cols + j])))));
      }
    }
  } else {
    fprintf(stderr, "Error: Invalid activation function..");
    exit(1);
  }
}

float MSELoss(Matrix y, Matrix y_hat) {
  if (y.rows != y_hat.rows || y.cols != y_hat.cols) {
    fprintf(
        stderr,
        "Error: Unable to calculate MSE Loss between different sized matrices: "
        "(%" PRIu64 " x %" PRIu64 ") and (%" PRIu64 " x %" PRIu64 ")\n",
        y.rows, y.cols, y_hat.rows, y_hat.cols);
    exit(1);
  }

  float cost = 0.0;

  for (uint64_t i = 0; i < y.rows * y.cols; i++) {
    cost += pow(y.data[i] - y_hat.data[i], 2);
  }

  cost /= (float)y.rows;

  return cost;
}

float crossEntropyLoss(Matrix y, Matrix y_hat) {
  float loss = 0.0;
  uint64_t examples = y.rows;
  uint64_t outputs = y.cols;

  for (uint64_t i = 0; i < examples; i++) {
    for (uint64_t k = 0; k < outputs; k++) {
      float current_y = y.data[i * y.cols + k];

      if (current_y == 0)
        continue;

      float current_y_hat = y_hat.data[i * y_hat.cols + k];
      loss -= current_y * log(current_y_hat + 1e-9f);
    }
  }

  loss /= examples;
  return loss;
}

// Helper

void getHighestIndexes(Matrix outputs, Matrix *result) {
  if (outputs.rows != result->rows) {
    fprintf(stderr,
            "Error: Unable to get highest indices with wrong result matrix "
            "size: Outputs = (%" PRIu64 " x %" PRIu64 "), result* = (%" PRIu64
            " x %" PRIu64 ")",
            outputs.rows, outputs.cols, result->rows, result->cols);
    exit(1);
  }

  if (result->cols != 1) {
    fprintf(
        stderr,
        "Error: List of highest indexes must have cols = 1: result* = (%" PRIu64
        " x %" PRIu64 ")",
        result->rows, result->cols);
    exit(1);
  }

  for (uint64_t i = 0; i < outputs.rows; i++) {
    uint64_t max_index = 0;
    float max_value = outputs.data[i * outputs.cols];

    for (uint64_t j = 0; j < outputs.cols; j++) {
      if (outputs.data[i * outputs.cols + j] > max_value) {
        max_index = j;
        max_value = outputs.data[i * outputs.cols + j];
      }
    }
    result->data[i] = max_index;
  }
}
