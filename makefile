# Compiler
CXX = g++
CC = gcc

# Flags
DEPFLAGS = -MMD -MP
OPT = -O0
CXXFLAGS = -std=c++17 -Icore/include -Iexamples/include $(DEPFLAGS) $(OPT)
CFLAGS = -Icore/include -Iexamples/include $(DEPFLAGS) $(OPT)

# Directories
CORE_SRC_DIR = core/src
CORE_INC_DIR = core/include
TEST_DIR = tests
EXAMPLE_DIR = examples/src
BUILD_DIR = build

# Source files
CXX_SRC_FILES = $(wildcard $(CORE_SRC_DIR)/*.cpp) $(wildcard $(EXAMPLE_DIR)/*.cpp) $(wildcard $(TEST_DIR)/*.cpp)
C_SRC_FILES = $(wildcard $(CORE_SRC_DIR)/*.c) $(wildcard $(EXAMPLE_DIR)/*.c) $(wildcard $(TEST_DIR)/*.c)

# Object files (in build directory with same base name as source files)
CXX_OBJ_FILES = $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(notdir $(CXX_SRC_FILES)))
C_OBJ_FILES = $(patsubst %.c, $(BUILD_DIR)/%.o, $(notdir $(C_SRC_FILES)))

# Dependency files
DEP_FILES = $(CXX_OBJ_FILES:.o=.d) $(C_OBJ_FILES:.o=.d)

# Output binaries
CORE_TARGET = $(BUILD_DIR)/core
TEST_TARGET = $(BUILD_DIR)/test
EXAMPLE_TARGET = $(BUILD_DIR)/example

# Include dependency files
-include $(DEP_FILES)

# Default target
all: $(CORE_TARGET) $(TEST_TARGET) $(EXAMPLE_TARGET)

# Linking the core library (C++)
$(CORE_TARGET): $(CXX_OBJ_FILES) $(C_OBJ_FILES)
	$(CXX) -o $@ $^

# Linking the test executable (C++)
$(TEST_TARGET): $(CXX_OBJ_FILES) $(C_OBJ_FILES)
	$(CXX) -o $@ $^

# Linking the example executable (C or C++)
$(EXAMPLE_TARGET): $(CXX_OBJ_FILES) $(C_OBJ_FILES)
	$(CXX) -o $@ $^

# Compile C++ source files to object files in the build directory
$(BUILD_DIR)/%.o: $(CORE_SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(EXAMPLE_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(TEST_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile C source files to object files in the build directory
$(BUILD_DIR)/%.o: $(CORE_SRC_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(EXAMPLE_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(TEST_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Create build directory if it does not exist
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Run the example program
run_example: $(EXAMPLE_TARGET)
	./$(EXAMPLE_TARGET)

# Run the tests
run_test: $(TEST_TARGET)
	./$(TEST_TARGET)

# Clean target
clean:
	rm -rf $(BUILD_DIR)

# Separate build targets
build_core: $(CORE_TARGET)
	@echo "Core library built successfully."

build_tests: $(TEST_TARGET)
	@echo "Test program built successfully."

build_example: $(EXAMPLE_TARGET)
	@echo "Example program built successfully."

.PHONY: all clean run_example run_test build_core build_tests build_example
