# tesseract++ - N-Dimensional Tensor Library in C++

## Overview
`TensorND` is a versatile C++ library for handling N-dimensional tensors. This library is templated, allowing efficient and dynamic tensor operations for scientific computing, deep learning, and other applications requiring high-dimensional data manipulation. The `TensorND` class is optimized for mathematical operations, supports tensor arithmetic, slicing, and provides an intuitive interface for various tensor transformations.

## Features
- **N-dimensional Support**: Supports tensors with any number of dimensions specified at compile-time.
- **Arithmetic Operations**: Element-wise addition, subtraction, multiplication, and division with scalars or other tensors.
- **Index-based Access**: Allows flexible access with variadic index operators and array-based indexing.
- **Transpose and Shape Manipulation**: Easily transpose tensors and retrieve shape information.
- **Utility Functions**: Generate identity tensors, set diagonal elements, initialize tensors to zero or random values.
- **Einsum-style Tensor Contraction**: Efficient contraction for performing complex tensor multiplications.
- **Memory Efficiency**: Supports copy and move constructors for efficient memory management.
  
## Installation
To use `TensorND`, simply include the `TensorND.h` header in your project and ensure that your compiler supports C++17 or later.

```cpp
#include "TensorND.h"
```

## Usage
To create a tensor, specify its data type and dimensions. For example, here’s a 2D tensor of size 3x3:

```cpp
TensorND<float, 3, 3> tensor;
```

### Basic Initialization and Operations
You can initialize a tensor with a specific value, access elements with indices, and perform arithmetic operations.

```cpp
// Initialize all elements to a specific value
TensorND<int, 2, 2> tensor(5);  // 2x2 tensor with all elements set to 5

// Accessing and modifying elements
tensor(0, 1) = 10;               // Set element at (0,1) to 10

// Adding a scalar to each element
auto tensor2 = tensor + 3;       // Each element incremented by 3

// Element-wise tensor addition
TensorND<int, 2, 2> result = tensor + tensor2;
```

## Examples
Here are a few examples of how to use `TensorND` to perform common tensor operations:

### Transposing a Matrix
For a 2D tensor (matrix), you can transpose it easily:

```cpp
TensorND<float, 2, 3> matrix;

// Initialize matrix values

matrix.transpose(); // Only for 2D tensors
std::cout << "Shape: " << matrix.getShape();  // Get tensor shape
```

### Setting the Tensor to an Identity Matrix
You can quickly initialize a tensor as an identity matrix if it’s 2D and square.

```cpp
auto identity = TensorND<float, 3, 3>::I();
```

### Performing Element-Wise Multiplication
TensorND allows for element-wise operations between tensors of the same size:

```cpp
TensorND<int, 2, 2> tensorA, tensorB;
// Fill tensorA and tensorB with values
TensorND<int, 2, 2> product = tensorA * tensorB;
```

### Utility Functions

Setting Values:

```cpp
tensor.setToZero();                           // Set all elements to 0
tensor.setIdentity();                         // Set as identity matrix
```

Random Initialization:

```cpp
tensor.setRandom(10, -10);                  // Random values between -10 and 10
```

Diagonal Elements:

```cpp
tensor.setDiagonal(1.0);                    // Set diagonal elements to 1.0
```

### Printing Tensors

Print 2D, 3D, or 4D tensors:

```cpp
tensor2D.print();                               // Prints 2D tensor
tensor3D.print();                             // Prints 3D tensor
tensor4D.print();                             // Prints 4D tensor
```

## How to run tests

It is recommended to run the tests to ensure that the library is working correctly. To run the tests, simply run:

```bash
make -j 20 run_test 
```
