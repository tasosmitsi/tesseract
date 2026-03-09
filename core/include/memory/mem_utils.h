#pragma once

#include "config.h"
#include "simple_type_traits.h"

/**
 * @file mem_utils.h
 * @brief STL-free memory utilities.
 */

namespace detail
{

    /**
     * @brief Fill a block of memory with a byte value.
     *
     * Uses `__builtin_memset` when available, otherwise falls back
     * to a byte-wise loop.
     *
     * @param ptr   Destination pointer.
     * @param value Byte value to set (only lowest 8 bits used).
     * @param size  Number of bytes to fill.
     */
    inline void memset(void *ptr, int value, my_size_t size) noexcept
    {
#if defined(__has_builtin) && __has_builtin(__builtin_memset)
        __builtin_memset(ptr, value, size);
#else
        auto *p = static_cast<unsigned char *>(ptr);
        for (my_size_t i = 0; i < size; ++i)
            p[i] = static_cast<unsigned char>(value);
#endif
    }

    /**
     * @brief Copy a block of memory.
     *
     * Uses `__builtin_memcpy` when available, otherwise falls back
     * to a byte-wise loop. Source and destination must not overlap.
     *
     * @param dst  Destination pointer.
     * @param src  Source pointer.
     * @param size Number of bytes to copy.
     */
    inline void memcpy(void *dst, const void *src, my_size_t size) noexcept
    {
#if defined(__has_builtin) && __has_builtin(__builtin_memcpy)
        __builtin_memcpy(dst, src, size);
#else
        auto *d = static_cast<unsigned char *>(dst);
        const auto *s = static_cast<const unsigned char *>(src);
        for (my_size_t i = 0; i < size; ++i)
            d[i] = s[i];
#endif
    }

} // namespace detail

/**
 * @brief Fill a contiguous buffer with a given value.
 *
 * For POD types initialized to zero, delegates to detail::memset
 * for optimal codegen. Falls back to a scalar loop otherwise.
 *
 * @tparam T Element type.
 * @param ptr   Pointer to the start of the buffer.
 * @param count Number of elements to fill.
 * @param value Value to assign to each element.
 */
template <typename T>
inline void fill_n_optimized(T *ptr, my_size_t count, const T &value)
{
    if constexpr (is_pod_v<T>)
    {
        if (value == T{})
        {
            detail::memset(ptr, 0, count * sizeof(T));
            return;
        }
    }

    for (my_size_t i = 0; i < count; ++i)
    {
        ptr[i] = value;
    }
}

/**
 * @brief Copy elements from a source buffer to a destination buffer.
 *
 * For POD types, delegates to detail::memcpy for optimal codegen.
 * Falls back to an element-wise loop otherwise.
 *
 * @tparam T Element type.
 * @param src   Pointer to the source buffer.
 * @param dst   Pointer to the destination buffer.
 * @param count Number of elements to copy.
 */
template <typename T>
inline void copy_n_optimized(const T *src, T *dst, my_size_t count)
{
    if constexpr (is_pod_v<T>)
    {
        detail::memcpy(dst, src, count * sizeof(T));
        return;
    }

    for (my_size_t i = 0; i < count; ++i)
    {
        dst[i] = src[i];
    }
}