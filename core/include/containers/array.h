#pragma once

#include "config.h" // for my_size_t

/**
 * @brief Fixed-size array container for embedded systems.
 *
 * @tparam T    Element type
 * @tparam N    Number of elements (compile-time constant)
 *
 * Drop-in replacement for std::array without STL dependency.
 * All operations are constexpr for compile-time use.
 *
 * Usage:
 *   constexpr Array<int, 3> arr = {1, 2, 3};
 *   static_assert(arr[0] == 1);
 *   static_assert(arr.size() == 3);
 */
template <typename T, my_size_t N>
struct Array
{
    // ========================================================================
    // DATA
    // ========================================================================

    T data[N];

    // ========================================================================
    // ELEMENT ACCESS
    // ========================================================================

    FORCE_INLINE constexpr T &operator[](my_size_t i) noexcept
    {
        return data[i];
    }

    FORCE_INLINE constexpr const T &operator[](my_size_t i) const noexcept
    {
        return data[i];
    }

    constexpr T &at(my_size_t i)
    {
        if (i >= N)
            MyErrorHandler::error("Array::at: index out of bounds");
        return data[i];
    }

    constexpr const T &at(my_size_t i) const
    {
        if (i >= N)
            MyErrorHandler::error("Array::at: index out of bounds");
        return data[i];
    }

    FORCE_INLINE constexpr T &front() noexcept { return data[0]; }
    FORCE_INLINE constexpr const T &front() const noexcept { return data[0]; }

    FORCE_INLINE constexpr T &back() noexcept { return data[N - 1]; }
    FORCE_INLINE constexpr const T &back() const noexcept { return data[N - 1]; }

    // ========================================================================
    // CAPACITY
    // ========================================================================

    FORCE_INLINE static constexpr my_size_t size() noexcept { return N; }
    FORCE_INLINE static constexpr bool empty() noexcept { return N == 0; }

    // ========================================================================
    // ITERATORS
    // ========================================================================

    FORCE_INLINE constexpr T *begin() noexcept { return data; }
    FORCE_INLINE constexpr const T *begin() const noexcept { return data; }

    FORCE_INLINE constexpr T *end() noexcept { return data + N; }
    FORCE_INLINE constexpr const T *end() const noexcept { return data + N; }

    // ========================================================================
    // OPERATIONS
    // ========================================================================

    constexpr void fill(const T &value) noexcept
    {
        for (my_size_t i = 0; i < N; ++i)
            data[i] = value;
    }

    constexpr void swap(Array &other) noexcept
    {
        for (my_size_t i = 0; i < N; ++i)
        {
            T tmp = data[i];
            data[i] = other.data[i];
            other.data[i] = tmp;
        }
    }
};

// ============================================================================
// SPECIALIZATION FOR EMPTY ARRAY (N=0)
// ============================================================================

/**
 * @brief Empty array specialization.
 *
 * Useful for generic code that may produce zero-sized arrays.
 * Most accessors are omitted since no elements exist.
 */
template <typename T>
struct Array<T, 0>
{
    // No data member — zero-size C arrays are non-standard

    // ========================================================================
    // ELEMENT ACCESS (limited - no elements exist)
    // ========================================================================

#ifdef TESSERACT_BOUNDS_CHECK
    constexpr T &at(my_size_t)
    {
        MyErrorHandler::error("Array<T,0>::at: empty array has no elements");
    }

    constexpr const T &at(my_size_t) const
    {
        MyErrorHandler::error("Array<T,0>::at: empty array has no elements");
    }
#endif

    // operator[], front(), back() intentionally omitted — undefined for empty array

    // ========================================================================
    // CAPACITY
    // ========================================================================

    FORCE_INLINE static constexpr my_size_t size() noexcept { return 0; }
    FORCE_INLINE static constexpr bool empty() noexcept { return true; }

    // ========================================================================
    // ITERATORS
    // ========================================================================

    FORCE_INLINE constexpr T *begin() noexcept { return nullptr; }
    FORCE_INLINE constexpr const T *begin() const noexcept { return nullptr; }

    FORCE_INLINE constexpr T *end() noexcept { return nullptr; }
    FORCE_INLINE constexpr const T *end() const noexcept { return nullptr; }

    // ========================================================================
    // OPERATIONS
    // ========================================================================

    constexpr void fill(const T &) noexcept { /* nothing to fill */ }
    constexpr void swap(Array &) noexcept { /* nothing to swap */ }
};

// ============================================================================
// DEDUCTION GUIDE (C++17)
// ============================================================================

/**
 * Enables brace-initialization without explicit template arguments:
 *   Array arr = {1, 2, 3};  // deduces Array<int, 3>
 *
 * First argument determines element type T.
 * Total argument count determines size N.
 */
template <typename T, typename... U>
Array(T, U...) -> Array<T, 1 + sizeof...(U)>;
