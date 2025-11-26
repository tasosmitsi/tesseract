# FusedTensorND – N-Dimensional Tensor Library in C++

## Overview

**FusedTensorND** is a high-performance C++17-compatible tensor library supporting **N-dimensional fixed-size tensors** with compile-time shape definitions. It is ideal for scientific computing, linear algebra, deep learning, and any application that requires efficient, statically-sized tensor operations.

This library emphasizes **zero-overhead abstraction**, **expression fusion**, and **compile-time shape enforcement** while providing a familiar and intuitive interface for tensor manipulation.

---

## Features

### ✅ N-Dimensional Static Tensors
- Tensors with dimensions defined at compile time: `FusedTensorND<T, D1, D2, ..., Dn>`
- Efficient memory layout for embedded and performance-critical applications

### ✅ Arithmetic Operations
- Element-wise operations: `+`, `-`, `*`, `/` with tensors and scalars
- Unary negation (`-tensor`)
- Fused expression evaluation to reduce intermediate allocations

### ✅ Index-Based Access
- Access individual elements using `tensor(i, j, k, ...)`
- Bounds checking in debug mode (if enabled)

### ✅ Transpose and Shape Manipulation
- In-place and non-in-place transpose for 2D and N-D tensors
- Support for arbitrary axis permutations
- `getShape()`, `getNumDims()`, `getDim(dim)` for shape inspection

### ✅ Utility Initialization
- `setIdentity()` – Set diagonal elements to 1 (for square 2D tensors)
- `setDiagonal(value)` – Set diagonal to a constant
- `setToZero()` – Fill all elements with 0
- `setHomogen(value)` – Fill all elements with the same value
- `setSequencial()` – Fill elements with sequential values
- `setRandom(min, max)` – Random values in range

### ✅ Tensor Comparison
- `==` and `!=` operators
- Dimension checks included – mismatched tensors throw on comparison

### ✅ Transpose-Safe Operations
- Transposed tensors participate correctly in arithmetic and equality

### ✅ Expression Fusion Support
- Deep expression fusion like `a + b * c + d` is supported and optimized
- Avoids creation of intermediate temporaries

### ✅ Einsum-style Contractions
- Perform tensor contractions along specified axes
- Example: matrix multiplication via contraction

### ✅ Copy and Move Semantics
- Efficient assignment and resource reuse

---

## Example Usage

### Create a 2D tensor

```cpp
FusedTensorND<float, 3, 3> mat;
