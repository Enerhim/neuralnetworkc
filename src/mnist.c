#include "../include/mnist.h"
#include "../include/math.h"
// Changing the endian from big to small

uint32_t read_u32_be(FILE *f) {
  uint8_t b[4];
  fread(b, 1, 4, f);
  return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3];
}

Matrix *load_mnist_dataset(const char *path, uint64_t *noImages) {
  FILE *f = fopen(path, "rb");

  if (!f) {
    fprintf(stderr, "Error: Unable to open file %s", path);
    return NULL;
  }

  uint32_t magic = read_u32_be(f);
  if (magic != 0x00000803) {
    fprintf(stderr, "Invalid Image File\n");
    fclose(f);
    return NULL;
  }

  uint32_t count_ = read_u32_be(f);
  uint32_t rows = read_u32_be(f);
  uint32_t cols = read_u32_be(f);

  Matrix *images = malloc(sizeof(Matrix) * count_);

  uint8_t buffer[784];

  for (uint64_t i = 0; i < count_; i++) {
    images[i] = createMatrix(rows, cols);

    fread(buffer, 1, rows * cols, f);

    for (uint32_t j = 0; j < rows * cols; j++) {
      images[i].data[j] = buffer[j] / 255.0;
    }
  }

  fclose(f);
  *noImages = count_;
  return images;
}

Matrix load_mnist_labels(const char *path, uint64_t *out_count) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "Error: Unable to open file %s", path);
    exit(1);
  }

  if (read_u32_be(f) != 0x00000801) {
    fprintf(stderr, "Invalid MNIST label file\n");
    fclose(f);
    exit(1);
  }

  uint32_t count = read_u32_be(f);
  Matrix labels = createMatrix(count, 1);

  fread(&labels.data, 1, count, f);

  fclose(f);
  *out_count = count;
  return labels;
}
