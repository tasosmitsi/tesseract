#pragma once
#include <cstring> // for std::memset
#include "config.h"
#include "simple_type_traits.h"

template <typename T>
inline void fill_n_optimized(T *ptr, my_size_t count, const T &value)
{
    if constexpr (is_pod_v<T>)
    {
        if (value == T{})
        {
            std::memset(ptr, 0, count * sizeof(T));
            return;
        }
    }

    for (my_size_t i = 0; i < count; ++i)
    {
        ptr[i] = value;
    }
}
