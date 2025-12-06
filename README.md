<div align="center">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="docs/tesseract_logo.png">
  <img alt="tesseract logo" src="docs/tesseract_logo_reversed.png" width="85%">
</picture>

tesseract: The ultimate N-dimensional tensor library in C++, embedded systems optimized.

<h3>

[Homepage](https://github.com/tasosmitsi/tesseract) | [Documentation]() | [Discord]()

</h3>

</div>

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
FusedMatrix long operations                    100             1    38.6383 ms 
                                        379.222 us    378.281 us    382.076 us 
                                        7.50839 us    2.09985 us    16.1349 us 
                                                                               
Eigen long operations                          100             1    44.1392 ms 
                                        390.483 us    388.091 us    393.515 us 
                                        13.7401 us    11.7672 us     17.857 us 
                                                                               
FusedMatrix matmul                             100             1    139.517 ms 
                                        1.42035 ms    1.38667 ms    1.45643 ms 
                                        178.058 us    161.304 us    198.055 us 
                                                                               
Eigen matmul                                   100             1     3.6168 ms 
                                        33.2281 us    32.7464 us    34.1266 us 
                                        3.26674 us      2.147 us    5.13524 us 
                                                                               
FusedMatrix inverse                            100             2     2.9798 ms 
                                        11.2536 us    11.1814 us    11.3924 us 
                                        493.664 ns     296.14 ns    787.344 ns 
                                                                               
Eigen inverse                                  100            24     1.9224 ms 
                                        803.661 ns    799.698 ns     811.35 ns 
                                        27.2454 ns    15.8709 ns     40.933 ns 
                                                                               
FusedMatrix Cholesky Decomposition             100             8     2.0376 ms 
                                        2.13216 us    2.12418 us    2.14682 us 
                                        53.6205 ns    31.6454 ns    78.9504 ns 
                                                                               
Eigen Cholesky Decomposition                   100             6     2.0094 ms 
                                        3.34483 us    3.33556 us    3.36331 us 
                                        63.4971 ns    36.7604 ns    99.5144 ns 
                                                                               

-------------------------------------------------------------------------------
Benchmarks - float
-------------------------------------------------------------------------------
benchmark name                       samples       iterations    est run time
                                     mean          low mean      high mean
                                     std dev       low std dev   high std dev
-------------------------------------------------------------------------------
FusedMatrix long operations                    100             1    19.9591 ms 
                                        189.859 us    189.251 us    192.129 us 
                                        5.31992 us    1.34706 us    12.2927 us 
                                                                               
Eigen long operations                          100             1    20.6471 ms 
                                         191.33 us    190.681 us    193.514 us 
                                        5.46045 us    1.84318 us    12.3329 us 
                                                                               
FusedMatrix matmul                             100             1     136.07 ms 
                                        1.41467 ms    1.38643 ms    1.44283 ms 
                                        144.019 us    131.382 us    159.118 us 
                                                                               
Eigen matmul                                   100             2     3.6582 ms 
                                        16.6498 us    16.4148 us    17.0552 us 
                                        1.53392 us    1.01504 us    2.13999 us 
                                                                               
FusedMatrix inverse                            100             2     2.7164 ms 
                                        10.3433 us    10.3094 us    10.4102 us 
                                        232.874 ns    129.883 ns    360.298 ns 
                                                                               
Eigen inverse                                  100            25      1.865 ms 
                                        748.007 ns    744.573 ns    755.046 ns 
                                        24.0746 ns    13.8936 ns    37.7925 ns 
                                                                               
FusedMatrix Cholesky Decomposition             100             8     2.1192 ms 
                                        1.94345 us    1.93748 us    1.95652 us 
                                        42.7868 ns    21.8059 ns    75.2195 ns 
                                                                               
Eigen Cholesky Decomposition                   100            11     1.9811 ms 
                                        1.79071 us    1.78585 us    1.80051 us 
                                        33.3831 ns     19.352 ns    52.6223 ns 
```
