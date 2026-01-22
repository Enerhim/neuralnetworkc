#include "../include/mnist.h"
#include "../include/math.h"
// Changing the endian from big to small

uint32_t read_u32_be(FILE *f) {
  uint8_t b[4];
  fread(b, 1, 4, f);
  return (b[0] << 24) | (b[1] << 16) | (b[2] << 8) | b[3];
}

Matrix load_mnist_dataset(const char *path, uint64_t *noImages, uint32_t size) {
  FILE *f = fopen(path, "rb");

  if (!f) {
    fprintf(stderr, "Error: Unable to open file %s", path);
    exit(1);
  }

  uint32_t magic = read_u32_be(f);
  if (magic != 0x00000803) {
    fprintf(stderr, "Invalid Image File\n");
    fclose(f);
    exit(1);
  }

  uint32_t count_ = read_u32_be(f);
  uint32_t rows = read_u32_be(f);
  uint32_t cols = read_u32_be(f);

  if (count_ < size) {
    fprintf(stderr, "Error: number of demanded elements greater than elements");
    exit(1);
  }

  Matrix images = createMatrix(size, rows * cols);
  uint8_t *buffer = (uint8_t *)malloc(rows * size * sizeof(uint8_t));
  if (!buffer) {
    fprintf(stderr, "Error: Failed to allocate image buffer of size %" PRIu8,
            size);
    exit(1);
  }

  for (uint64_t i = 0; i < size; i++) {
    fread(buffer, 1, rows * cols, f);
    for (uint32_t j = 0; j < rows * cols; j++) {
      images.data[i * cols * rows + j] = buffer[j] / 255.0;
    }
  }

  free(buffer);
  fclose(f);
  *noImages = count_;
  return images;
}

Matrix load_mnist_labels(const char *path, uint64_t *out_count, uint32_t size) {
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

  uint32_t count_ = read_u32_be(f);

  if (count_ < size) {
    fprintf(stderr, "Error: number of demanded elements greater than elements");
    exit(1);
  }

  Matrix labels = createMatrix(size, 1);
  uint8_t *buffer = (uint8_t *)malloc(size * sizeof(uint8_t));
  if (!buffer) {
    fprintf(stderr, "Error: Failed to allocate label buffer of size %" PRIu8,
            size);
    exit(1);
  }

  fread(buffer, 1, size, f);

  for (uint64_t i = 0; i < size; i++)
    labels.data[i] = (float)buffer[i];

  free(buffer);
  fclose(f);
  *out_count = count_;
  return labels;
}
