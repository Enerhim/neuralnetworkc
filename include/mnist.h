#pragma once
#include "math.h"

uint32_t read_u32_be(FILE *f);
Matrix load_mnist_dataset(const char *path, uint64_t *noImages, uint32_t size);
Matrix load_mnist_labels(const char *path, uint64_t *out_count, uint32_t size);
Matrix mnist_hot_encode(Matrix y);
float calculate_mnist_accuracy(Matrix y_hat, Matrix y);
