#include <inttypes.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define MIN(a, b) (a < b ? a : b)
#define MAX(a, b) (a > b ? a : b)

typedef struct {
  float *data;
  uint64_t rows, cols;
} Matrix;

Matrix createMatrix(uint64_t rows, uint64_t cols);
void copyMatrix(Matrix original, Matrix *target);

void fillMatrix(Matrix *A, float fill_val);

void addMatrices(Matrix A, Matrix B, Matrix *result);
void subtractMatrices(Matrix A, Matrix B, Matrix *result);
void mulMatrices(Matrix A, Matrix B, Matrix *result);
void freeMatrix(Matrix *A);

void printMatrix(Matrix A);

void transposeMatrix(Matrix A, Matrix *result);

void reluMatrix(Matrix *A);
void sigmoidMatrix(Matrix *A);
void linearMatrix(Matrix *A);
void softmaxMatrix(Matrix *A);
