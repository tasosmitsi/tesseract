#pragma once
#include <cstring> // for std::memset
#include "config.h"
#include "simple_type_traits.h"

template <typename T>
inline void copy_n_optimized(const T *src, T *dst, my_size_t count)
{
    if constexpr (is_pod_v<T>)
    {
        std::memcpy(dst, src, count * sizeof(T));
        return;
    }

    for (my_size_t i = 0; i < count; ++i)
    {
        dst[i] = src[i];
    }
}
