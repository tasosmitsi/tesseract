# Tesseract Naming Conventions

## Reserved (Never Use)
- `_CapitalLetter` — reserved for implementation
- `__anything` — reserved for implementation
- `_anything` in global scope — reserved

---

## Types

| Kind | Convention | Example |
|------|------------|---------|
| Classes / Structs | `PascalCase` | `Tensor`, `ExpressionTree`, `QuaternionBase` |
| Template parameters (type) | `PascalCase` | `typename Scalar`, `typename Derived` |
| Template parameters (non-type) | `PascalCase` or `snake_case` | `size_t Rank` or `size_t rank` |
| Type aliases (`using`) | `PascalCase` | `using ValueType = T;` |
| Concepts (C++20) | `PascalCase` | `concept TensorExpression` |
| Enums | `PascalCase` | `enum class StorageOrder` |
| Enum values | `PascalCase` | `StorageOrder::RowMajor` |

---

## Variables & Functions

| Kind | Convention | Example |
|------|------------|---------|
| Local variables | `snake_case` | `row_index`, `temp_buffer` |
| Function parameters | `snake_case` | `void resize(size_t new_size)` |
| Member variables (private) | `snake_case_` | `data_`, `shape_`, `strides_` |
| Member variables (public) | `snake_case` | `size`, `rank` |
| Static members | `snake_case_` | `default_allocator_` |
| Global constants | `kPascalCase` | `kMaxRank`, `kDefaultAlignment` |
| Constexpr variables | `kPascalCase` | `constexpr size_t kCacheLineSize = 64;` |
| Functions / methods | `snake_case` | `compute_strides()`, `at()` |

---

## Macros (avoid if possible)

| Kind | Convention | Example |
|------|------------|---------|
| Macros | `TESSERACT_SCREAMING_CASE` | `TESSERACT_ASSERT`, `TESSERACT_SIMD_ENABLED` |

Always prefix with `TESSERACT_` to avoid collisions.

---

## Namespaces

| Kind | Convention | Example |
|------|------------|---------|
| Namespaces | `snake_case` | `tesseract`, `tesseract::simd`, `tesseract::detail` |

- `detail` or `internal` — implementation details not for public use

---

## Files

| Kind | Convention | Example |
|------|------------|---------|
| Headers | `snake_case.hpp` | `tensor.hpp`, `expression_tree.hpp` |
| Implementation | `snake_case.cpp` | `tensor.cpp` |
| Template impl (if separate) | `snake_case.inl` | `tensor.inl` |

---

## Internal vs Public API

```cpp
namespace tesseract {

// Public API — documented, stable
template<typename Scalar, size_t Rank>
class Tensor { ... };

namespace detail {
    // Internal — may change without notice
    template<typename Expr>
    struct ExpressionEvaluator { ... };
}

} // namespace tesseract
```

---

## Examples

```cpp
namespace tesseract {

constexpr size_t kMaxRank = 8;

template<typename Scalar, size_t Rank>
class Tensor {
public:
    using ValueType = Scalar;
    using SizeType = size_t;

    Tensor() = default;
    explicit Tensor(std::array<size_t, Rank> shape);

    Scalar& at(size_t index);
    size_t size() const { return size_; }
    
    void reshape(std::array<size_t, Rank> new_shape);

private:
    Scalar* data_ = nullptr;
    size_t size_ = 0;
    std::array<size_t, Rank> shape_{};
    std::array<size_t, Rank> strides_{};
};

} // namespace tesseract
```

---

## Rationale

- **`snake_case` for functions**: Matches STL style, feels natural in C++
- **`PascalCase` for types**: Clear visual distinction from variables/functions
- **Trailing underscore for members**: Avoids `this->` clutter, distinguishes from locals
- **`k` prefix for constants**: Clear intent, avoids macro-style ALL_CAPS for non-macros
- **`detail` namespace**: Clear signal that internals are not stable API