#pragma once
#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MIN(a, b) (a < b ? a : b)
#define MAX(a, b) (a > b ? a : b)

typedef struct {
  float *data;
  uint64_t rows, cols;
} Matrix;

typedef void (*ActivationFunction)(Matrix *A);

Matrix createMatrix(uint64_t rows, uint64_t cols);
Matrix createRandomMatrix(uint64_t rows, uint64_t cols);
Matrix createXavierMatrix(uint64_t rows, uint64_t cols, uint64_t fanIn,
                          uint64_t fanOut);
Matrix createHeMatrix(uint64_t rows, uint64_t cols, uint64_t fanIn);
void copyMatrix(Matrix original, Matrix *target);

void fillMatrix(Matrix *A, float fill_val);

void addMatrices(Matrix A, Matrix B, Matrix *result);
void subtractMatrices(Matrix A, Matrix B, Matrix *result);
void mulMatrices(Matrix A, Matrix B, Matrix *result);
void scaleMatrix(Matrix A, float scalar, Matrix *result);
void extendVector(Matrix A, uint64_t n, Matrix *result);
void hadamardProduct(Matrix A, Matrix B, Matrix *result);

void freeMatrix(Matrix *A);

void printMatrix(Matrix A);
void printShape(Matrix A);
Matrix getRow(Matrix A, uint64_t row_index);
Matrix getColumn(Matrix A, uint64_t col_index);

void transposeMatrix(Matrix A, Matrix *result);

void relu(Matrix *A);
void sigmoid(Matrix *A);
void linear(Matrix *A);
void softmax(Matrix *A);

void getHighestIndexes(Matrix outputs, Matrix *result);
float MSELoss(Matrix y, Matrix y_hat);
float crossEntropyLoss(Matrix y, Matrix y_hat);

void derivativeRelu(Matrix *A);
void derivativeLinear(Matrix *A);
void derivativeSigmoid(Matrix *A);
void derivativeSoftmax(Matrix *A);

void ActivationDerivative(ActivationFunction activation, Matrix *A);
