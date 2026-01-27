CC = gcc
CFLAGS = -Wall -Wextra -Wpedantic -std=c11 -Iinclude
OPTFLAGS = -O3 -march=native -ffast-math
DEBUGFLAGS = -g -O0 -DDEBUG -fsanitize=address -fsanitize=undefined
LDFLAGS = -lm

SRC_DIR = src
INC_DIR = include
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj

SOURCES = $(wildcard $(SRC_DIR)/*.c) main.c
OBJECTS = $(SOURCES:%.c=$(OBJ_DIR)/%.o)
DEPS = $(wildcard $(INC_DIR)/*.h)

TARGET = $(BUILD_DIR)/neural_net

all: CFLAGS += $(OPTFLAGS)
all: directories $(TARGET)

# Debug build
debug: CFLAGS += $(DEBUGFLAGS)
debug: LDFLAGS += -fsanitize=address -fsanitize=undefined
debug: directories $(TARGET)

# Create necessary directories
directories:
	@mkdir -p $(OBJ_DIR)/$(SRC_DIR)
	@mkdir -p $(BUILD_DIR)

# Link object files to create executable
$(TARGET): $(OBJECTS)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)
	@echo "Build complete: $(TARGET)"

# Compile source files to object files
$(OBJ_DIR)/%.o: %.c $(DEPS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# Run the program
run: all
	./$(TARGET)

# Run with debug build
debug-run: debug
	./$(TARGET)

# Profile build
profile: CFLAGS += $(OPTFLAGS) -pg
profile: LDFLAGS += -pg
profile: directories $(TARGET)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)
	rm -f gmon.out

# Clean and rebuild
rebuild: clean all

# Show compiler and flag information
info:
	@echo "Compiler: $(CC)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "OPTFLAGS: $(OPTFLAGS)"
	@echo "LDFLAGS: $(LDFLAGS)"
	@echo "Sources: $(SOURCES)"
	@echo "Objects: $(OBJECTS)"

# Help target
help:
	@echo "Available targets:"
	@echo "  all        - Build optimized release version (default)"
	@echo "  debug      - Build with debug symbols and sanitizers"
	@echo "  run        - Build and run release version"
	@echo "  debug-run  - Build and run debug version"
	@echo "  profile    - Build with profiling support"
	@echo "  clean      - Remove build artifacts"
	@echo "  rebuild    - Clean and rebuild"
	@echo "  info       - Show build configuration"
	@echo "  help       - Show this help message"

.PHONY: all debug directories run debug-run profile clean rebuild info help
