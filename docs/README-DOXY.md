\mainpage notitle

<div align="center">

![tesseract logo](tesseract_logo_reversed.png)
<br>
<br>
<br>
The ultimate N-dimensional tensor library in C++, embedded systems optimized.

<h3>

[Homepage](https://github.com/tasosmitsi/tesseract) | [Documentation](https://tasosmitsi.github.io/tesseract) | [Discord]

</h3>

</div>

[![Documentation](https://img.shields.io/badge/docs-doxygen-blue.svg)](https://tasosmitsi.github.io/tesseract/)

---

## Overview
Tesseract is a versatile C++ library for handling N-dimensional tensors. This library is templated, allowing efficient static and dynamic tensor operations for embedded systems, scientific computing, deep learning, and other applications requiring high-dimensional data manipulation. The `TensorND` class is optimized for mathematical operations, supports tensor arithmetic, slicing, and provides an intuitive interface for various tensor transformations.

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

### Tensor Arithmetic

Addition, Subtraction, Multiplication, and Division:

```cpp
TensorND<double, 3, 3> mat1, mat2;
mat1.setIdentity();
mat2.setIdentity();

auto addition = mat1 + mat2;       // Tensor addition
auto subtraction = mat1 - mat2;    // Tensor subtraction
auto multiplication = mat1 * mat2; // Tensor multiplication
auto scalarDivision = mat1 / 2.0;  // Divide by a scalar
```

### Transposing a Tensor
For a 2D tensor (matrix), you can transpose it easily:

```cpp
TensorND<float, 2, 3> matrix;

// Initialize matrix values

matrix.transpose(True); // Only for 2D tensors, true for in-place transpose
// or
auto transposed = matrix.transpose(); // Transpose and return a new tensor (not in-place)

std::cout << "Shape: " << matrix.getShape();  // Get tensor shape
```

For a higher-dimensional tensor, you can permute the axes:

```cpp
TensorND<float, 2, 3, 4> tensor;

// Initialize tensor values

tensor.transpose([1, 2, 0], true); // Permute axes, true for in-place transpose
// or
auto permuted_tensor = tensor.transpose([1, 2, 0]); // Permute axes and return a new tensor (not in-place)

std::cout << "Shape: " << tensor.getShape();  // Get tensor shape
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
tensor.setIdentity();                         // Set as identity
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

### Equality and Assignment

Compare Tensors:

```cpp
TensorND<double, 3, 3> mat1, mat2;
mat1.setIdentity();
mat2.setIdentity();

if (mat1 == mat2) {
    std::cout << "Tensors are equal!" << std::endl;
}

// or

if (mat1 != mat2) {
    std::cout << "Tensors are not equal!" << std::endl;
}
```

Assign One Tensor to Another:

```cpp
mat2 = mat1; // Assign mat1 to mat2
```

### Tensor Contraction (Einsum-style)

Perform tensor contraction using the `einsum` function:

```cpp
```

## How to run tests

It is recommended to run the tests to ensure that the library is working correctly. To run the tests, simply run:

```bash
make -j 20 run_test
```

## Benchmarks

The following benchmarks compare the performance of `FusedMatrix` operations against `Eigen` library operations for both double and float data types. These test are executed on single-threaded mode to provide a fair comparison of the core computational efficiency of each library. Moreover, AVX2 optimizations are enabled to leverage SIMD capabilities for enhanced performance in both libraries.

```
Benchmarks - double
-------------------------------------------------------------------------------

benchmark name                       samples       iterations    est run time
                                     mean          low mean      high mean
                                     std dev       low std dev   high std dev
-------------------------------------------------------------------------------
FusedMatrix long operations                    100             1     37.716 ms 
                                        377.387 us    376.776 us    379.341 us 
                                        5.03923 us    1.71825 us    11.2186 us 
                                                                               
Eigen long operations                          100             1    39.5535 ms 
                                        383.382 us    382.775 us    385.385 us 
                                        5.04039 us    1.54613 us    11.2372 us 
                                                                               
FusedMatrix matmul                             100             1     50.801 ms 
                                        548.602 us    530.716 us    567.471 us 
                                         93.448 us    84.2679 us    106.142 us 
                                                                               
Eigen matmul                                   100             1     3.4708 ms 
                                        33.0388 us    32.5843 us     34.137 us 
                                        3.37313 us     1.7275 us    6.41646 us 
                                                                               
FusedMatrix inverse                            100             2     2.1964 ms 
                                        9.22072 us    9.16082 us     9.3357 us 
                                        406.015 ns    245.778 ns    636.794 ns 
                                                                               
Eigen inverse                                  100            24       1.92 ms 
                                        804.018 ns    800.209 ns     811.26 ns 
                                        25.7764 ns    15.4741 ns    39.7277 ns 
                                                                               
FusedMatrix Cholesky Decomposition             100             5     1.8955 ms 
                                         3.1683 us    3.15744 us    3.19468 us 
                                         83.607 ns    39.4999 ns    144.428 ns 
                                                                               
Eigen Cholesky Decomposition                   100             6     2.0226 ms 
                                        3.36218 us    3.35291 us    3.38386 us 
                                        68.4079 ns    34.4429 ns    118.772 ns 
                                                                               

-------------------------------------------------------------------------------
Benchmarks - float
-------------------------------------------------------------------------------
benchmark name                       samples       iterations    est run time
                                     mean          low mean      high mean
                                     std dev       low std dev   high std dev
-------------------------------------------------------------------------------
FusedMatrix long operations                    100             1    18.9086 ms 
                                        188.852 us    188.035 us    191.978 us 
                                        7.39626 us    1.33233 us    17.3781 us 
                                                                               
Eigen long operations                          100             1    20.5788 ms 
                                        191.755 us    190.711 us    195.182 us 
                                        8.54103 us    1.91308 us    18.7689 us 
                                                                               
FusedMatrix matmul                             100             1    57.0981 ms 
                                         572.09 us    554.786 us    593.386 us 
                                        98.0738 us    82.1576 us     114.87 us 
                                                                               
Eigen matmul                                   100             2     3.5762 ms 
                                         16.602 us    16.4049 us    16.9708 us 
                                        1.33158 us    785.436 ns    1.98529 us 
                                                                               
FusedMatrix inverse                            100             2     2.1618 ms 
                                        9.05416 us    8.98452 us    9.17101 us 
                                        452.552 ns    290.792 ns    646.619 ns 
                                                                               
Eigen inverse                                  100            25     1.8625 ms 
                                        749.428 ns    745.386 ns    757.463 ns 
                                        27.9641 ns    15.0259 ns    45.3381 ns 
                                                                               
FusedMatrix Cholesky Decomposition             100             6       2.13 ms 
                                        2.92102 us    2.91226 us    2.93821 us 
                                        60.2316 ns    34.3328 ns    95.5562 ns 
                                                                               
Eigen Cholesky Decomposition                   100            11     1.9712 ms 
                                        1.79068 us    1.78596 us    1.80027 us 
                                        32.5313 ns    18.8424 ns    50.5642 ns /                
```
